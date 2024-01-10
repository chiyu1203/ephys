import time, os, json, warnings
import spikeinterface.full as si
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import probeinterface as pi
from probeinterface.plotting import plot_probe
import numcodecs
warnings.simplefilter("ignore")
#oe_folder = Path('/data_local/DataSpikeSorting/SI_tutorial_cambridgeneurotech_2023')
# oe_folder = Path("/home/alessio/Documents/data/spiketutorials/Official_Tutorial_SI_0.99_Nov23/")
#oe_folder = oe_folder / 'openephys_recording/2023-08-23_15-56-05'
def main(thisDir, json_file):
    oe_folder=Path(thisDir)
    #oe_folder=Path(r"C:\Users\neuroLaptop\Documents\Open Ephys\P-series-32channels\GN00002")
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    if analysis_methods.get("load_sorter_file")==True and (oe_folder / "results_sc2").is_dir():
        sorted_spikes = si.read_sorter_folder(oe_folder/"results_sc2")
    else:
        if analysis_methods.get("load_prepocessed_file")==True and (oe_folder / "preprocessed").is_dir():
            rec_of_interest = si.load_extractor(oe_folder / "preprocessed")
        else:
            full_raw_rec = si.read_openephys(oe_folder)
            fs = full_raw_rec.get_sampling_frequency()
            trace_snippet = full_raw_rec.get_traces(start_frame=int(fs*0), end_frame=int(fs*2))

            #load probe information
            manufacturer = 'cambridgeneurotech'
            probe_name = 'ASSY-37-P-2'
            probe = pi.get_probe(manufacturer, probe_name)
            print(probe)

            probe.wiring_to_device('ASSY-116>RHD2132')
            probe.to_dataframe(complete=True).loc[:, ["contact_ids", "shank_ids", "device_channel_indices"]]
            raw_rec = full_raw_rec.set_probe(probe)
            probe_rec = raw_rec.get_probe()
            probe_rec.to_dataframe(complete=True).loc[:, ["contact_ids", "device_channel_indices"]]


            raw_rec.annotate(description="Dataset for SI tutorial")
            #preprocessing
            recording_f = si.bandpass_filter(raw_rec, freq_min=300, freq_max=9000)
            recording_cmr = si.common_reference(recording_f, reference='global', operator='median')

            bad_channel_ids, channel_labels = si.detect_bad_channels(recording_f, method='coherence+psd')
            print('bad_channel_ids', bad_channel_ids)
            print('channel_labels', channel_labels)

            recording_good_channels_f = recording_f.remove_channels(bad_channel_ids)
            recording_good_channels = si.common_reference(recording_good_channels_f, reference='global', operator='median')

        if analysis_methods.get("analyse_entire_recording") ==True:
            rec_of_interest = recording_good_channels
        else:
            fs = recording_cmr.get_sampling_frequency()
            rec_of_interest = recording_good_channels.frame_slice(start_frame=0*fs, end_frame=100*fs)
            #raw_rec_sub = raw_rec.frame_slice(start_frame=0*fs, end_frame=300*fs)

        n_cpus = os.cpu_count()
        n_jobs = n_cpus - 4
        job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)
        compressor_name="zstd"
        if (oe_folder / "preprocessed_compressed.zarr").is_dir():
            if analysis_methods.get("save_prepocessed_file")==True and analysis_methods.get("overwrite_curated_dataset")==True:
                compressor = numcodecs.Blosc(cname="zstd", clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)
                recording_saved = rec_of_interest.save(format="zarr", folder=oe_folder / "preprocessed_compressed.zarr",
                                                    compressor=compressor,
                                                    **job_kwargs)
                print(f"Overwrite existing file with compressor: {compressor_name}")
            else:
                print("Skip saving this file.")
        else:
            compressor = numcodecs.Blosc(cname=compressor_name, clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)
            recording_saved = rec_of_interest.save(format="zarr", folder=oe_folder / "preprocessed_compressed.zarr",
                                                    compressor=compressor,
                                                    **job_kwargs)
            print(f"First time to save this file. Testing compressor: {compressor_name}")
        

        






        '''
        if (oe_folder / "preprocessed").is_dir():
            recording_saved = si.load_extractor(oe_folder / "preprocessed")
        else:
            recording_saved = recording_sub.save(folder=oe_folder / "preprocessed", **job_kwargs)

        recording_loaded = si.load_extractor(oe_folder / "preprocessed")

        if (oe_folder / "preprocessed_compressed.zarr").is_dir():
            recording_saved = si.read_zarr(oe_folder / "preprocessed_compressed.zarr")
        else:
            compressor = numcodecs.Blosc(cname="zstd", clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)
            recording_saved = recording_sub.save(format="zarr", folder=oe_folder / "preprocessed_compressed.zarr",
                                                compressor=compressor,
                                                **job_kwargs)
        '''

        # run spike sorting on entire recording
        sorted_spikes = si.run_sorter(sorter_name=analysis_methods.get("sorter_name"), recording=rec_of_interest, remove_existing_folder=True,
                                    output_folder=oe_folder / 'results_tc2',
                                    verbose=True)
    '''
    for unit in sorting_tc2.get_unit_ids():
    print(f'Spike train of a unit:{sorting_tc2.get_unit_spike_train(unit_id=unit)}')
    '''
    return sorted_spikes

        #print(f'Spike train of a unit: {sorted_spikes.get_unit_spike_train(unit_id=1)}')
        #print(f'Spike train of a unit (in s): {sorted_spikes.get_unit_spike_train(unit_id=1, return_times=True)}')

if __name__ == "__main__":
    #thisDir = r"C:\Users\neuroLaptop\Documents\Open Ephys\P-series-32channels\GN00003\2023-12-28_14-39-40"
    thisDir = r"C:\Users\neuroLaptop\Documents\Open Ephys\2023-06-06_22-15-37"
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    sorted_spikes=main(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")