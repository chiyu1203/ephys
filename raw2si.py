import time, os, json, warnings
import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.sorters as ss
import spikeinterface.qualitymetrics as sq
import spikeinterface.exporters as sep
#import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
#import numpy as np
from pathlib import Path
import probeinterface as pi
from probeinterface.plotting import plot_probe
import numcodecs
warnings.simplefilter("ignore")
import spikeinterface.curation as scur

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
        sorter_suffix="_KS3"
    elif this_sorter.lower() == "kilosort4":
        sorter_suffix="_KS4"
    result_folder_name="results"+sorter_suffix
    sorting_folder_name="sorting"+sorter_suffix
    n_cpus = os.cpu_count()
    n_jobs = n_cpus - 4
    job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)
    

    if analysis_methods.get("load_sorting_file")==True:
        if (oe_folder / result_folder_name).is_dir():
            sorting_spikes = ss.read_sorter_folder(oe_folder/result_folder_name)
            #sorting_spikes = si.core.load_extractor(oe_folder/sorting_folder_name)#this acts quite similar than above one line.
        else:
            print(f"no result folder found for {sorter_suffix} sorter")
        #load recording in case there is a need to extract waveform    
        if (oe_folder / "preprocessed_compressed.zarr").is_dir():
            recording_saved = si.read_zarr(oe_folder / "preprocessed_compressed.zarr")
        elif (oe_folder / "preprocessed").is_dir():
            recording_saved = si.load_extractor(oe_folder / "preprocessed")
        else:
            print(f"no pre-processed folder found. Unable to extract waveform")
            return sorting_spikes
    else:
        if analysis_methods.get("load_prepocessed_file")==True and (oe_folder / "preprocessed_compressed.zarr").is_dir():
            recording_saved = si.read_zarr(oe_folder / "preprocessed_compressed.zarr")
            fs = recording_saved.get_sampling_frequency()
        elif analysis_methods.get("load_prepocessed_file")==True and (oe_folder / "preprocessed").is_dir():
            print("Looks like you do not have compressed files. Read the original instead")
            recording_saved = si.load_extractor(oe_folder / "preprocessed")
            fs = recording_saved.get_sampling_frequency()
        else:
            print("Load meta information from openEphys")
            full_raw_rec = se.read_openephys(oe_folder,load_sync_timestamps=True)
            # To show the start of recording time 
            # full_raw_rec.get_times()[0]
            event=se.read_openephys_event(oe_folder)
            #event_channel_ids=channel_ids
            #events = event.get_events(channel_id=channel_ids[1], segment_index=0)# a complete record of events including [('time', '<f8'), ('duration', '<f8'), ('label', '<U100')]
            events_times=event.get_event_times(channel_id=event.channel_ids[1],segment_index=0)# this record ON phase of sync pulse
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
            rec_of_interest.annotate(is_filtered=True)#needed to add this because loading saved-preprocessed data is not labeled as filtered data
        if analysis_methods.get("analyse_entire_recording") ==False:
            start_sec=1
            end_sec=900
            rec_of_interest = rec_of_interest.frame_slice(start_frame=start_sec*fs, end_frame=end_sec*fs)#need to check if I can do this online
        # ideally only saving preprocessed files in compressed format
        compressor_name="zstd"
        if (oe_folder / "preprocessed_compressed.zarr").is_dir():
            if analysis_methods.get("save_prepocessed_file")==True and analysis_methods.get("overwrite_curated_dataset")==True:
                compressor = numcodecs.Blosc(cname="zstd", clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE)
                recording_saved = rec_of_interest.save(format="zarr", folder=oe_folder / "preprocessed_compressed.zarr",
                                                    compressor=compressor,overwrite=True,
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
            if this_sorter == "kilosort3":
                kilosort_3_path = r'C:\Users\neuroPC\Documents\GitHub\Kilosort-3.0.2'
                ss.Kilosort3Sorter.set_kilosort3_path(kilosort_3_path)
                sorter_params = {'do_correction': False}
            else:
                print("use kilosort4")
                sorter_params={'dminx': 250,'nearest_templates':10}
            sorting_spikes = ss.run_sorter(sorter_name=this_sorter, recording=recording_saved, remove_existing_folder=True,
                                    output_folder=oe_folder / result_folder_name,
                                    verbose=True, **sorter_params)
            #sorter_params.update({"projection_threshold": [9, 9]})##this is a parameters from Christopher Michael Jernigan's experiences with Wasps
        else:
            motion_folder=oe_folder / "motion"
            rec_correct_motion=spre.correct_motion(recording=recording_saved, preset="kilosort_like", folder=motion_folder)
            motion_info = spre.load_motion_info(motion_folder)
            recording_saved=rec_correct_motion
            sorter_params=ss.get_default_sorter_params(this_sorter)

        # run spike sorting on recording of interest            
            sorting_spikes = ss.run_sorter(sorter_name=this_sorter, recording=recording_saved, remove_existing_folder=True,
                                        output_folder=oe_folder / result_folder_name,
                                        verbose=True)
        ##this will return a sorting object
    
    w_rs=sw.plot_rasters(sorting_spikes, time_range=(0,30),backend="matplotlib")
    if analysis_methods.get("save_sorting_file")==True and analysis_methods.get("overwrite_curated_dataset")==True:
        sorting_loaded_spikes=sorting_spikes.save(folder=oe_folder / sorting_folder_name, overwrite=True)  
    for unit in sorting_spikes.get_unit_ids():
        print(f'with {this_sorter} sorter, Spike train of a unit:{sorting_spikes.get_unit_spike_train(unit_id=unit)}')
    
    ##extracting waveform
    # the extracted waveform based on sparser signals (channels) makes the extraction faster. 
    # However, if the channels are not dense enough the right waveform can not be properly extracted.
    sorting_wout_excess_spikes = scur.remove_excess_spikes(sorting_spikes, recording_saved)
    sorting_spikes=sorting_wout_excess_spikes    
    if analysis_methods.get("extract_waveform_sparse")==True:
        waveform_folder_name="waveforms_sparse"+sorter_suffix
        we = si.extract_waveforms(recording_saved, sorting_spikes, folder=oe_folder / waveform_folder_name, 
                          sparse=True, overwrite=True,**job_kwargs)
    else:
        waveform_folder_name="waveforms_dense"+sorter_suffix
        we = si.extract_waveforms(recording_saved, sorting_spikes, folder=oe_folder / waveform_folder_name, 
                          sparse=False, overwrite=True, **job_kwargs)
        sparsity = si.compute_sparsity(we, method='radius', radius_um=100.0)


    ##curation
    # the safest way to curate spikes are manual curation, which phy seems to be a good package to deal with that
    # When exporting spikes data to phy, amplitudes and pc features can also be calculated    
    # if you do not wish to use phy, we can calculate quality metrics with other packages in spikeinterface.
    ##exporting to phy 
    ##still need to check whether methods are used to compute pc features and amplitude when exporting to phy, and whether
    ## we want those methods to be default methods. 
    ## If not, we should get some spost methods before this step and turn the two computer options in export_to_phy off. 
    if analysis_methods.get("export_to_phy")==True and analysis_methods.get("overwrite_existing_phy")==True:
        phy_folder_name="phy"+sorter_suffix
        sep.export_to_phy(we, output_folder=oe_folder / phy_folder_name, 
                    compute_amplitudes=True, compute_pc_features=True, copy_binary=False,remove_if_exists=True,
                    **job_kwargs)
    else:
        pc = spost.compute_principal_components(we, n_components=3, load_if_exists=False, **job_kwargs)
        all_labels, all_pcs = pc.get_all_projections()
        print(f"All PC scores shape: {all_pcs.shape}")
        #we.get_available_extension_names()
        #pc = we.load_extension("principal_components")
        #all_labels, all_pcs = pc.get_data()
        amplitudes = spost.compute_spike_amplitudes(we, outputs="by_unit", load_if_exists=True, **job_kwargs)
        unit_locations = spost.compute_unit_locations(we, method="monopolar_triangulation", load_if_exists=True)
        spike_locations = spost.compute_spike_locations(we, method="center_of_mass", load_if_exists=True, **job_kwargs)
        ccgs, bins = spost.compute_correlograms(we)
        similarity = spost.compute_template_similarity(we)
        qm_params = sq.get_default_qm_params()
        metric_names = sq.get_quality_metric_list()
        qm = sq.compute_quality_metrics(we, metric_names=metric_names, verbose=True,  qm_params=qm_params, **job_kwargs)
        print(qm)
        
    ##outputing a report
    if analysis_methods.get("export_report")==True:
        report_folder_name="report"+sorter_suffix
        sep.export_report(we, output_folder=oe_folder / report_folder_name)
    return recording_saved,sorting_spikes

if __name__ == "__main__":
    #thisDir = r"C:\Users\neuroLaptop\Documents\Open Ephys\P-series-32channels\GN00003\2023-12-28_14-39-40"
    #thisDir = r"Z:\DATA\experiment_openEphys\P-series-32channels\2024-02-01_18-55-51"
    thisDir = r"C:\Users\neuroPC\Documents\Open Ephys\2024-02-01_15-25-25"
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    recording_saved,sorting_spikes=main(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
