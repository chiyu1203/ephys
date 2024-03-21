import time, os, json, warnings
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
#import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
#import numpy as np
from pathlib import Path
import probeinterface as pi
from probeinterface.plotting import plot_probe
import numcodecs
warnings.simplefilter("ignore")

def main(thisDir, json_file):
    oe_folder=Path(thisDir)
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    this_sorter=analysis_methods.get("sorter_name")
    this_experimenter=analysis_methods.get("experimenter")
    if this_sorter.lower() == "spykingcircus2":
        sorter_suffix="_SC2"
    elif this_sorter.lower() == "kilosort3":
        sorter_suffix="_KC3"
    result_folder_name="results"+sorter_suffix
    sorting_folder_name="sorting"+sorter_suffix

    if analysis_methods.get("load_sorting_file")==True:
        if (oe_folder / result_folder_name).is_dir():
            sorting_spikes = ss.read_sorter_folder(oe_folder/result_folder_name)
            #sorting_spikes = si.core.load_extractor(oe_folder/sorting_folder_name)#this acts quite similar than above one line.
        else:
            print(f"no result folder found for {sorter_suffix} sorter")
    else:
        if analysis_methods.get("load_prepocessed_file")==True and (oe_folder / "preprocessed_compressed.zarr").is_dir():
            recording_saved = si.read_zarr(oe_folder / "preprocessed_compressed.zarr")
            fs = recording_saved.get_sampling_frequency()
        elif analysis_methods.get("load_prepocessed_file")==True and (oe_folder / "preprocessed").is_dir():
            print("Looks like you do not have compressed files. Read the original instead")
            recording_saved = si.core.load_extractor(oe_folder / "preprocessed")
            fs = recording_saved.get_sampling_frequency()
        else:
            print("Load meta information from openEphys")
            full_raw_rec = se.read_openephys(oe_folder)
            fs = full_raw_rec.get_sampling_frequency()
            if analysis_methods.get("load_raw_traces")==True:
                trace_snippet = full_raw_rec.get_traces(start_frame=int(fs*0), end_frame=int(fs*2))

            #load probe information
            manufacturer = 'cambridgeneurotech'
            probe_name = 'ASSY-37-P-2'
            probe = pi.get_probe(manufacturer, probe_name)
            print(probe)

            probe.wiring_to_device('ASSY-116>RHD2132')
            probe.to_dataframe(complete=True).loc[:, ["contact_ids", "shank_ids", "device_channel_indices"]]
            #drop AUX channels here
            raw_rec = full_raw_rec.set_probe(probe)
            probe_rec = raw_rec.get_probe()
            probe_rec.to_dataframe(complete=True).loc[:, ["contact_ids", "device_channel_indices"]]


            raw_rec.annotate(description=f"Dataset of {this_experimenter}")#should change here for something related in the future
            #preprocessing
            #apply band pass filter
            recording_f = spre.bandpass_filter(raw_rec, freq_min=300, freq_max=9000)
            #apply common median reference
            if analysis_methods.get("analyse_good_channels_only")==True:
                '''
                This step should be done before saving preprocessed files because ideally the preprocessed file we want to create is something ready for spiking
                detection, which means neural traces gone through bandpass filter and common reference. 
                However, applying common reference takes signals from channels of interest which requires us to decide what we want to do with other bad or noisy channels first.
                '''
                bad_channel_ids, channel_labels = spre.detect_bad_channels(recording_f, method='coherence+psd')
                print('bad_channel_ids', bad_channel_ids)
                print('channel_labels', channel_labels)

                recording_f = recording_f.remove_channels(bad_channel_ids)#need to check if I can do this online
            
            recording_cmr = spre.common_reference(recording_f, reference='global', operator='median')
        if 'recording_cmr' in locals():
            rec_of_interest = recording_cmr
        else:
            rec_of_interest = recording_saved
        if analysis_methods.get("analyse_entire_recording") ==False:
            rec_of_interest = rec_of_interest.frame_slice(start_frame=0*fs, end_frame=100*fs)#need to check if I can do this online

        n_cpus = os.cpu_count()
        n_jobs = n_cpus - 4
        job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)
        # ideally only saving preprocessed files in compressed format
        compressor_name="zstd"
        if (oe_folder / "preprocessed_compressed.zarr").is_dir():
            if analysis_methods.get("save_prepocessed_file")==True and analysis_methods.get("overwrite_curated_dataset")==True:
                compressor = numcodecs.Blosc(cname="zstd", clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)
                recording_saved = rec_of_interest.save(format="zarr", folder=oe_folder / "preprocessed_compressed.zarr",
                                                    compressor=compressor,
                                                    **job_kwargs)
                print(f"Overwrite existing file with compressor: {compressor_name}")
            else:
                recording_saved = rec_of_interest
                print("Skip saving this Recording object and converting it to a binary file. Please make sure your sorters are happy with that")
        else:
            compressor = numcodecs.Blosc(cname=compressor_name, clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)
            recording_saved = rec_of_interest.save(format="zarr", folder=oe_folder / "preprocessed_compressed.zarr",
                                                    compressor=compressor,
                                                    **job_kwargs)
            print(f"First time to save this file. Testing compressor: {compressor_name}")

        # update parameters of sorters. For non-kilosort sorters, here is an additional step to correct motion artifact.
        if this_sorter.startswith("kilosort"):
            sorter_params=ss.get_default_sorter_params(this_sorter)
            sorter_params.update({"projection_threshold": [9, 9]})##this is a parameters from Christopher Michael Jernigan's experiences with Wasps
        else:
            motion_folder=oe_folder / "motion"
            rec_correct_motion=spre.correct_motion(recording=recording_saved, preset="kilosort_like", folder=motion_folder)
            motion_info = spre.load_motion_info(motion_folder)
            recording_saved=rec_correct_motion
            sorter_params=ss.get_default_sorter_params(this_sorter)

        # run spike sorting on recording of interest            
        sorting_spikes = ss.run_sorter(sorter_name=this_sorter, recording=recording_saved, remove_existing_folder=True,
                                    output_folder=oe_folder / result_folder_name,
                                    verbose=True, **sorter_params)
        
        ##this will return a sorting object
    
    w_rs=sw.plot_rasters(sorting_spikes, time_range=(0,30),backend="matplotlib")
    if analysis_methods.get("save_sorting_file")==True and analysis_methods.get("overwrite_curated_dataset")==True:
        sorting_loaded_spikes=sorting_spikes.save(folder=oe_folder / sorting_folder_name)  
    for unit in sorting_spikes.get_unit_ids():
        print(f'with {this_sorter} sorter, Spike train of a unit:{sorting_spikes.get_unit_spike_train(unit_id=unit)}')

    return sorting_spikes

if __name__ == "__main__":
    #thisDir = r"C:\Users\neuroLaptop\Documents\Open Ephys\P-series-32channels\GN00003\2023-12-28_14-39-40"
    thisDir = r"C:\Users\neuroLaptop\Documents\Open Ephys\2024-02-01_15-25-25"
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    sorting_spikes=main(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")