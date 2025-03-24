import time, os, json, warnings
import probeinterface as pi
from probeinterface.plotting import plot_probe
import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.sorters as ss
import spikeinterface.qualitymetrics as sq
import spikeinterface.exporters as sep
from open_ephys.analysis import Session
from estimate_drift_motion import AP_band_drift_estimation, LFP_band_drift_estimation
# from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
# from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording
from spikeinterface.sortingcomponents.motion import (
    correct_motion_on_peaks,
    interpolate_motion,
    estimate_motion,
)
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import numcodecs

##from brainbox.population.decode import get_spike_counts_in_bins
##from brainbox.task.trials import get_event_aligned_raster, get_psth
##from brainbox.ephys_plots import scatter_raster_plot
warnings.simplefilter("ignore")
import spikeinterface.curation as scur

"""
This pipeline uses spikeinterface as a backbone. This file includes preprocessing and sorting, converting raw data from openEphys to putative spikes by various sorters
"""


def generate_sorter_suffix(this_sorter):
    if this_sorter.lower() == "spykingcircus2":
        sorter_suffix = "_SC2"
    elif this_sorter.lower() == "kilosort3":
        sorter_suffix = "_KS3"
    elif this_sorter.lower() == "kilosort4":
        sorter_suffix = "_KS4"
    return sorter_suffix

def raw2si(thisDir, json_file):
    oe_folder = Path(thisDir)
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    this_sorter = analysis_methods.get("sorter_name")
    this_experimenter = analysis_methods.get("experimenter")
    probe_type = analysis_methods.get("probe_type")
    motion_corrector = analysis_methods.get("motion_corrector")
    sorter_suffix = generate_sorter_suffix(this_sorter)
    result_folder_name = "results" + sorter_suffix
    sorting_folder_name = "sorting" + sorter_suffix
    n_cpus = os.cpu_count()
    n_jobs = n_cpus - 4
    job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)

    if analysis_methods.get("load_sorting_file") == True:
        if (oe_folder / result_folder_name).is_dir():
            # sorting_spikes = ss.read_sorter_folder(oe_folder/result_folder_name)
            sorting_spikes = si.load_extractor(
                oe_folder / sorting_folder_name
            )  # this acts quite similar than above one line.
        else:
            print(f"no result folder found for {this_sorter}")
        # load recording in case there is a need to extract waveform
        if (oe_folder / "preprocessed_compressed.zarr").is_dir():
            recording_saved = si.read_zarr(oe_folder / "preprocessed_compressed.zarr")
            print(recording_saved.get_property_keys())
        elif (oe_folder / "preprocessed").is_dir():
            recording_saved = si.load_extractor(oe_folder / "preprocessed")
        else:
            print(f"no pre-processed folder found. Unable to extract waveform")
            return sorting_spikes
        recording_saved.annotate(is_filtered=True)
    else:
        if (
            analysis_methods.get("load_prepocessed_file") == True
            and (oe_folder / "preprocessed_compressed.zarr").is_dir()
        ):
            recording_saved = si.read_zarr(oe_folder / "preprocessed_compressed.zarr")
            print(recording_saved.get_property_keys())
            fs = recording_saved.get_sampling_frequency()
        elif (
            analysis_methods.get("load_prepocessed_file") == True
            and (oe_folder / "preprocessed").is_dir()
        ):
            print(
                "Looks like you do not have compressed files. Read the original instead"
            )
            recording_saved = si.load_extractor(oe_folder / "preprocessed")
            fs = recording_saved.get_sampling_frequency()
        else:
            print("Load meta information from openEphys")
            # session = Session(oe_folder)
            # recording = session.recordnodes[0].recordings[0]
            # camera_trigger_on_oe = recording.events.timestamp[
            #     (recording.events.line == 2) & (recording.events.state == 1)
            # ]
            # pd_on_oe = recording.events.timestamp[
            #     (recording.events.line == 1) & (recording.events.state == 1)
            # ]
            # pd_off_oe = recording.events.timestamp[
            #     (recording.events.line == 1) & (recording.events.state == 0)
            # ]
            raw_rec = se.read_openephys(oe_folder, load_sync_timestamps=True)
            # To show the start of recording time
            # raw_rec.get_times()[0]
            event = se.read_openephys_event(oe_folder)
            # event_channel_ids=channel_ids
            # events = event.get_events(channel_id=channel_ids[1], segment_index=0)# a complete record of events including [('time', '<f8'), ('duration', '<f8'), ('label', '<U100')]
            events_times = event.get_event_times(
                channel_id=event.channel_ids[1], segment_index=0
            )  # this record ON phase of sync pulse
            fs = raw_rec.get_sampling_frequency()
            if analysis_methods.get("load_raw_traces") == True:
                trace_snippet = raw_rec.get_traces(
                    start_frame=int(fs * 0), end_frame=int(fs * 2)
                )

            ################load probe information################
            if probe_type == "P2":
                manufacturer = "cambridgeneurotech"
                probe_name = "ASSY-37-P-2"
                probe = pi.get_probe(manufacturer, probe_name)
                print(probe)
                probe.wiring_to_device("ASSY-116>RHD2132")
                probe.to_dataframe(complete=True).loc[
                    :, ["contact_ids", "shank_ids", "device_channel_indices"]
                ]
            elif probe_type == "H10_stacked":
                stacked_probes = pi.read_probeinterface("H10_stacked_probes.json")
                probe = stacked_probes.probes[0]
            else:
                print("the name of probe not identified. stop the programme")
                return
            # drop AUX channels here
            raw_rec = raw_rec.set_probe(probe,group_mode='by_shank')
            probe_rec = raw_rec.get_probe()
            probe_rec.to_dataframe(complete=True).loc[
                :, ["contact_ids", "device_channel_indices"]
            ]

            raw_rec.annotate(
                description=f"Dataset of {this_experimenter}"
            )  # should change here for something related in the future
            ################ estimate motion with LFP band ################

            #As we do not analyse LFP data, there was no need to correct motion based on LFP band. However, this estimation can be good to cross-validate with spike-band based motion estimation
            # https://spikeinterface.readthedocs.io/en/latest/how_to/drift_with_lfp.html
            lfp_drift_estimation=False
            if lfp_drift_estimation:
                raw_rec_dict = raw_rec.split_by(property='group', outputs='dict')
                for group, rec_per_shank in raw_rec_dict.items():
                    LFP_band_drift_estimation(group,rec_per_shank,oe_folder)
            ################ preprocessing ################
            # apply band pass filter
            raw_rec = spre.astype(raw_rec, np.float32)
            recording_f = spre.bandpass_filter(raw_rec, freq_min=600, freq_max=6000,dtype="float32")
            # apply common median reference to remove common noise
            if analysis_methods.get("analyse_good_channels_only") == True:
                """
                This step should be done before saving preprocessed files because ideally the preprocessed file we want to create is something ready for spiking
                detection, which means neural traces gone through bandpass filter and common reference.
                However, applying common reference takes signals from channels of interest which requires us to decide what we want to do with other bad or noisy channels first.
                """
                bad_channel_ids, channel_labels = spre.detect_bad_channels(
                    recording_f, method="coherence+psd"
                )  # bad_channel_ids=np.array(['CH1','CH2','CH3','CH4','CH5','CH6','CH7','CH8','CH9','CH10','CH11','CH12','CH13','CH14','CH15','CH16'],dtype='<U64')
                #bad channel ids in ['CH2' 'CH22' 'CH25' 'CH27' 'CH29' 'CH30' 'CH31' 'CH32' 'CH33' 'CH34''CH35' 'CH36' 'CH37' 'CH38' 'CH39' 'CH41' 'CH43' 'CH80' 'CH112'] in 2025-03-19_18-02-13"
                print("bad_channel_ids", bad_channel_ids)
                print("channel_labels", channel_labels)

                recording_f = recording_f.remove_channels(
                    bad_channel_ids
                )  # need to check if I can do this online



            ##not sure if I should apply CAR by shank by shank
            recording_cmr = spre.common_reference(
                recording_f, reference="global", operator="median"
            )
        if "recording_cmr" in locals():
            rec_of_interest = recording_cmr
        else:
            rec_of_interest = recording_saved
            rec_of_interest.annotate(
                is_filtered=True
            )  # needed to add this somehow because when loading a preprocessed data saved in the past, that data would not be labeled as filtered data
        # Slice the recording if needed
        if analysis_methods.get("analyse_entire_recording") == False:
            start_sec = 1
            end_sec = 899
            rec_of_interest = rec_of_interest.frame_slice(
                start_frame=start_sec * fs, end_frame=end_sec * fs
            )
        # at the moment, leaving raw data intact while saving preprocessed files in compressed format but in the future,
        # we might want to remove the raw data to save space
        # more information about this idea can be found here https://github.com/SpikeInterface/spikeinterface/issues/2996#issuecomment-2486394230
        compressor_name = "zstd"
        if analysis_methods.get("save_prepocessed_file") == True:
            if (oe_folder / "preprocessed_compressed.zarr").is_dir():
                if analysis_methods.get("overwrite_curated_dataset") == True:
                    compressor = numcodecs.Blosc(
                        cname="zstd", clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE
                    )
                    recording_saved = rec_of_interest.save(
                        format="zarr",
                        folder=oe_folder / "preprocessed_compressed.zarr",
                        compressor=compressor,
                        overwrite=True,
                        **job_kwargs,
                    )
                    print(f"Overwrite existing file with compressor: {compressor_name}")
                else:
                    recording_saved = rec_of_interest
                    print(
                        "Skip saving this Recording object and converting it to a binary file. Please make sure your sorters are happy with that"
                    )
            else:
                compressor = numcodecs.Blosc(
                    cname=compressor_name, clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE
                )
                recording_saved = rec_of_interest.save(
                    format="zarr",
                    folder=oe_folder / "preprocessed_compressed.zarr",
                    compressor=compressor,
                    **job_kwargs,
                )
                print(
                    f"First time to save this file. Testing compressor: {compressor_name}"
                )
        else:
            recording_saved = rec_of_interest
            print(
                "Skip saving this Recording object and converting it to a binary file. Please make sure your sorters are happy with that"
            )

        ############################# correcting drift/motion ##########################
        # There are multiple ways to correct motion artifact. We can either use kilosort's motion correction or other methods like dredge.
        # kilosort's motion correction is part of its own preprocessing step after removing common noise with common median reference (CMR) or average reference (CAR).
        # Therefore, there are drift correction and whitening steps if kilosort or kilosort's motion correction is not used.
        # Motion correction method Dredge seems do well with shorter probe https://www.youtube.com/watch?v=RYKHoipT-2A
        # Note: if we use kilosort without using its own preprocessing steps, remember to turn off those steps when calling the function.
        # For more information, please refer to https://github.com/SpikeInterface/spikeinterface/issues/3483

        # split into a dict
        #recording_saved = spre.astype(recording_saved, "float")
        recordings_dict = recording_saved.split_by(property='group', outputs='dict')
        win_um=100
        sorter_params = ss.get_default_sorter_params(this_sorter)
        sorter_params.update({"dminx": 18.5,"nblocks": 0,"batch_size": 180000})
        sorter_params.update({"skip_kilosort_preprocessing": True})
        step_chan = 35
        recording_corrected_dict = {}
        for group, sub_recording in recordings_dict.items():
            print(len(sub_recording.ids_to_indices()))
            if len(sub_recording.ids_to_indices())<27:
                continue
            recording_corrected,_=AP_band_drift_estimation(group,sub_recording,oe_folder,analysis_methods,win_um,job_kwargs)
            #recording_corrected_dict[group]=recording_corrected
            rec_for_sorting = spre.whiten(
                recording=recording_corrected,
                mode="local",
                radius_um=step_chan * 2,
                dtype=float,
            )
            sorting_spikes = ss.run_sorter(
                sorter_name=this_sorter,
                recording=rec_for_sorting,
                remove_existing_folder=True,
                output_folder=oe_folder / result_folder_name,
                verbose=True,
                **sorter_params,
            )
            recording_corrected_dict[group]=sorting_spikes
        # if len(corrected_list)>1:
        #     recording_corrected = si.aggregate_channels(corrected_list)
        
        if motion_corrector =='testing':
            return print("drift/correction testing done")
        ############################# whitening ##########################
        elif motion_corrector =='kilosort':
        #use_kilosort_motion_correction = True
            rec_for_sorting = recording_corrected
        #if use_kilosort_motion_correction:
            print("use the default motion correction and whitening method in the kilosort")
            pass
        else:
            # apply whitening to remove spatial correction in the data
            step_chan = 35
            rec_for_sorting = spre.whiten(
                recording=recording_corrected,
                mode="local",
                radius_um=step_chan * 2,
                dtype=float,
            )

        ############################# spike sorting ##########################
        #print(f'theses sorters are installed in this PC {ss.installed_sorters()}')
        print(f"run spike sorting with {this_sorter}")
        sorter_params = ss.get_default_sorter_params(this_sorter)
        if this_sorter.startswith("kilosort"):
            #update parameters based on motion correction method
            if motion_corrector =='kilosort':
                pass
            else:
                sorter_params.update({"skip_kilosort_preprocessing": True})
            #update parameters based on probe type    
            if probe_type=='H10_stacked':
                sorter_params.update({"dminx": 18.5,"nblocks": 0,"batch_size": 180000})
            elif probe_type=='P2':
                sorter_params.update({"dminx": 22.5,"nearest_templates": 16, "max_channel_distance": 32,"nblocks": 0,"batch_size": 180000})
            #update parameters based on the version of kilosort
            if this_sorter == "kilosort3":
                kilosort_3_path = r"C:\Users\neuroPC\Documents\GitHub\Kilosort-3.0.2"
                ss.Kilosort3Sorter.set_kilosort3_path(kilosort_3_path)
                sorter_params.update({"do_correction": False})
            else:
                print("use kilosort4")
                # sorter_params={'dminx': 250,'nearest_templates':10}
            sorting_spikes = ss.run_sorter(
                sorter_name=this_sorter,
                recording=rec_for_sorting,
                remove_existing_folder=True,
                do_CAR=False,
                output_folder=oe_folder / result_folder_name,
                verbose=True,
                **sorter_params,
            )
            # sorter_params.update({"projection_threshold": [9, 9]})##this is a parameters from Christopher Michael Jernigan's experiences with Wasps
        else:
            sorting_spikes = ss.run_sorter(
                sorter_name=this_sorter,
                recording=rec_for_sorting,
                remove_existing_folder=True,
                output_folder=oe_folder / result_folder_name,
                verbose=True,
                job_kwargs=job_kwargs,
            )
        ##this will return a sorting object
    ############################# spike sorting preview and saving ##########################
        w_rs = sw.plot_rasters(sorting_spikes, time_range=(0, 30), backend="matplotlib")
        if (
            analysis_methods.get("save_sorting_file") == True
            and analysis_methods.get("overwrite_curated_dataset") == True
        ):
            sorting_spikes.save(folder=oe_folder / sorting_folder_name, overwrite=True)

    return print("Spiking sorting done. The rest of the tasks can be done in other PCs")
    # for unit in sorting_spikes.get_unit_ids():
    #     print(f'with {this_sorter} sorter, Spike train of a unit:{sorting_spikes.get_unit_spike_train(unit_id=unit)}')


if __name__ == "__main__":
    # thisDir = r"C:\Users\neuroLaptop\Documents\Open Ephys\P-series-32channels\GN00003\2023-12-28_14-39-40"
    # thisDir = r"Z:\DATA\experiment_openEphys\P-series-32channels\2024-02-01_15-25-25"
    # thisDir = r"C:\Users\neuroPC\Documents\Open Ephys\2024-05-01_17-39-51"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23015\240201\coherence\session1\2024-02-01_15-25-25"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23016\240201\coherence\session1\2024-02-01_18-55-51"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN24001\240529\coherence\session1\2024-05-29_15-33-31"
    # thisDir = r"D:\Open Ephys\2025-03-10_20-25-05"
    #thisDir = r"D:\Open Ephys\2025-03-19_18-02-13"
    #thisDir = r"Z:\DATA\experiment_openEphys\H-series-128channels\2025-03-23_20-47-26"
    #thisDir = r"Z:\DATA\experiment_openEphys\H-series-128channels\2025-03-23_21-33-38"
    thisDir = r"Z:\DATA\experiment_openEphys\H-series-128channels\2025-03-23_20-47-26"
    # thisDir = r"D:\Open Ephys\2025-02-23_20-39-04"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23019\240507\coherence\session1\2024-05-07_23-08-55"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23018\240422\coherence\session2\2024-04-22_01-09-50"
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    raw2si(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
