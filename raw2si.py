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
n_cpus = os.cpu_count()
n_jobs = n_cpus - 4

global_job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)
# global_job_kwargs = dict(n_jobs=16, chunk_duration="5s", progress_bar=False)
si.set_global_job_kwargs(**global_job_kwargs)

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
def motion_correction_shankbyshank(recording_saved,oe_folder,analysis_methods):
    win_um=100        
    recording_corrected_dict = {}
    #create a temporary boolean here to account for correct motion not ready to accept dict. If the recording is an Object, it wwill first split it based groups. If an recording Object has no the group attribute,
    #that means it does not go through this line raw_rec = raw_rec.set_probe(probe,group_mode='by_shank') to create the attribute. In this case, a fake Group0 is created just because the function needs that
    #After correcting motion, a dictionary will be created, which can be used in the following analysis
    if type(recording_saved) == dict:
        for group, sub_recording in recording_saved.items():
            print(f"this probe has number of channels to analyse: {len(sub_recording.ids_to_indices())}")
            recording_corrected,_=AP_band_drift_estimation(group,sub_recording,oe_folder,analysis_methods,win_um,global_job_kwargs)
            recording_corrected_dict[group]=recording_corrected
    elif len(np.unique(recording_saved.get_property('group')))>1:
        recording_saved = recording_saved.split_by(property='group', outputs='dict')
        for group, sub_recording in recording_saved.items():
            print(f"this probe has number of channels to analyse: {len(sub_recording.ids_to_indices())}")
            recording_corrected,_=AP_band_drift_estimation(group,sub_recording,oe_folder,analysis_methods,win_um,global_job_kwargs)
            recording_corrected_dict[group]=recording_corrected
    else:
        group=0
        recording_corrected,_=AP_band_drift_estimation(group,recording_saved,oe_folder,analysis_methods,win_um,global_job_kwargs)
        recording_corrected_dict[group]=recording_corrected
    return recording_corrected_dict

def get_preprocessed_recording(oe_folder,analysis_methods):
    tmin_tmax = analysis_methods.get("tmin_tmax")
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
        this_experimenter = analysis_methods.get("experimenter")
        probe_type = analysis_methods.get("probe_type")

        plot_traces = analysis_methods.get("plot_traces")
        if oe_folder.stem =='2025-03-05_13-45-15':
            raw_rec = se.read_openephys(oe_folder, load_sync_timestamps=True,block_index=0,stream_id='0')
        else:
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
        if probe_type == "H10_stacked":
            stacked_probes = pi.read_probeinterface("H10_stacked_probes_2D.json")
            probe = stacked_probes.probes[0]
        else:
            manufacturer = "cambridgeneurotech"
            if probe_type == "P2":
                probe_name = "ASSY-37-P-2"
                connector_type="ASSY-116>RHD2132"
            elif probe_type == "H5":
                probe_name = "ASSY-77-H5"
                connector_type="ASSY-77>Adpt.A64-Om32_2x-sm-NN>RHD2164"
            else:
                print("the name of probe not identified. stop the programme")
                return
            probe = pi.get_probe(manufacturer, probe_name)
            probe.to_dataframe(complete=True).loc[
                :, ["contact_ids", "shank_ids", "device_channel_indices"]
            ]
            probe.wiring_to_device(connector_type)
        print(probe)
        # drop AUX channels here
        #raw_rec = raw_rec.set_probe(probe,group_mode='by_shank')
        raw_rec = raw_rec.set_probe(probe,group_mode='by_shank')
        probe_rec = raw_rec.get_probe()
        probe_rec.to_dataframe(complete=True).loc[
            :, ["contact_ids", "device_channel_indices"]
        ]

        raw_rec.annotate(
            description=f"Dataset of {this_experimenter}"
        )  # should change here for something related in the future
        ################ estimate motion with LFP band ################

        #As we do not analyse LFP data, there was no need to correct motion based on LFP band. However, this estimation can be good to validate the result from spike-band based motion estimation
        # https://spikeinterface.readthedocs.io/en/latest/how_to/drift_with_lfp.html
        lfp_drift_estimation=False
        if lfp_drift_estimation:
            raw_rec_dict = raw_rec.split_by(property='group', outputs='dict')
            for group, rec_per_shank in raw_rec_dict.items():
                LFP_band_drift_estimation(group,rec_per_shank,oe_folder)
        



        ################ preprocessing ################
        # apply band pass filter
        ### need to double check whether there is a need to convert data type to float32. It seems that this will increase the size of the data
        recording_f = spre.bandpass_filter(raw_rec, freq_min=600, freq_max=6000,dtype="float32")# it sounds that people recommend to run two separate bandpass filter for motion estimation and for spike sorting.
        # recording_f = spre.highpass_filter(raw_rec, freq_min=300,dtype="float32")
        

        if analysis_methods.get("analyse_good_channels_only") == True:
            """
            This step should be done before saving preprocessed files because ideally the preprocessed file we want to create is something ready for spiking
            detection, which means neural traces gone through bandpass filter and common reference.
            However, applying common reference takes signals from channels of interest which requires us to decide what we want to do with other bad or noisy channels first.
            """
            bad_channel_ids, channel_labels = spre.detect_bad_channels(
                recording_f, method="coherence+psd"
            )
            print("bad_channel_ids", bad_channel_ids)
            print("channel_labels", channel_labels)

            recording_f = recording_f.remove_channels(
                bad_channel_ids
            )  #try this functino interpolate_bad_channels when I can put 3 shanks in the brain plus when there is some noisy channels
            #https://spikeinterface.readthedocs.io/en/stable/api.html#spikeinterface.preprocessing.interpolate_bad_channels

        ##start to split the recording into groups here because remove bad channels function is not ready to receive dict as input
        recordings_dict = recording_f.split_by(property='group', outputs='dict')
        if plot_traces:
            fig0=plt.figure()
            for group, rec_per_shank in recordings_dict.items():
                figcode=int(f"22{group+1}")
                ax=fig0.add_subplot(figcode)
                sw.plot_traces(rec_per_shank,  mode="auto",ax=ax)
            plt.show()
            #shankid=0
            #sw.plot_traces({f"shank{shankid+1}": recordings_dict[shankid]},  mode="auto",time_range=[10, 10.1], backend="ipywidgets")
            #sw.plot_traces(recordings_dict[shankid],  mode="auto",time_range=[10, 10.1])

        # apply common median reference to remove common noise
        # recording_cmr = spre.common_reference(
        #     recordings_dict, reference="global", operator="average"
        # )
        recording_cmr = spre.common_reference(
            recording_f, reference="global", operator="median"
        )
        # another filter to consider: https://github.com/SpikeInterface/SpikeInterface-Training-Edinburgh-May24/blob/main/hands_on/preprocessing/preprocessing.ipynb
        # recording_cmr = spre.highpass_spatial_filter(
        #     recording_f
        #)
    if "recording_cmr" in locals():
        rec_of_interest = recording_cmr
    else:
        rec_of_interest = recording_saved
        rec_of_interest.annotate(
            is_filtered=True
        )  # needed to add this somehow because when loading a preprocessed data saved in the past, that data would not be labeled as filtered data
    # Slice the recording if needed
    if tmin_tmax[1]>0 and tmin_tmax[1]>tmin_tmax[0]:
        start_sec = tmin_tmax[0]
        end_sec = tmin_tmax[1]
        if type(rec_of_interest)==dict:
            tmp = {}
            for group, sub_recording in rec_of_interest.items():
                tmp[group] = sub_recording.frame_slice(start_frame=start_sec * fs, end_frame=end_sec * fs)
            rec_of_interest=tmp
        else:
            rec_of_interest = rec_of_interest.frame_slice(start_frame=start_sec * fs, end_frame=end_sec * fs)
    elif tmin_tmax[1]<0:
        print("tmax <0 means to analyse the entire recording")
    else:
        ValueError("tmax needs to be bigger than tmin to select certain section of the recording")
    return rec_of_interest


def raw2si(thisDir, json_file):
    oe_folder = Path(thisDir)
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    this_sorter = analysis_methods.get("sorter_name")
    motion_corrector = analysis_methods.get("motion_corrector")
    plot_traces = analysis_methods.get("plot_traces")
    probe_type = analysis_methods.get("probe_type")
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
        rec_of_interest=get_preprocessed_recording(oe_folder,analysis_methods)
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
                    if type(rec_of_interest) == dict:#create a temporary boolean here to account for that save.() function can not take dict as input
                        rec_of_interest=si.aggregate_channels(rec_of_interest)
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
                if type(rec_of_interest) == dict:#create a temporary boolean here to account for that save.() function can not take dict as input
                    rec_of_interest=si.aggregate_channels(rec_of_interest)
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
        
        # use manual splitting for motion for now because correct_motion function does not take dict as input yet
        #the first two parameters to test in motion correction: "win_step_um" and "win_scale_um"
        recording_corrected_dict=motion_correction_shankbyshank(recording_saved,oe_folder,analysis_methods)

        if plot_traces:
            fig1=plt.figure()
            for group, rec_per_shank in recording_corrected_dict.items():
                figcode=int(f"22{group+1}")
                ax=fig1.add_subplot(figcode)
                sw.plot_traces(rec_per_shank,  mode="auto",ax=ax)
            plt.show()   

        if motion_corrector =='testing':
            return print("drift/correction testing is finished")
        ############################# whitening ##########################
        elif motion_corrector =='kilosort_default':
            #setting motion_corrector to 'kilosort_default' will skip motion_correction_shankbyshank
            if type(recording_saved)==dict:
                rec_for_sorting=si.aggregate_channels(recording_saved)
            else:
        #use_kilosort_motion_correction = True
                rec_for_sorting = recording_saved
        #if use_kilosort_motion_correction:
            print("use the default motion correction and whitening method in the kilosort")
            pass
        else:
            # create a temporary option here to account for manual splitting during motion correction
            if len(recording_corrected_dict)>1:
                recording_corrected=recording_corrected_dict
            else:
                recording_corrected=recording_corrected_dict[0]
            # fig0=plt.figure()
            # r_range=[50,100,200,400]
            # #recording_saved.channel_ids[:10]np.linspace(1,10,10,dtype=int)
            # i=0
            # for this_r in r_range:
            #     rec_w = spre.whiten(recording=recording_saved,mode="local",radius_um=this_r,int_scale=200,dtype=float)
            #     figcode=int(f"15{i+1}")
            #     ax=fig0.add_subplot(figcode)
            #     if i==3:
            #         figcode=int(f"15{i+2}")
            #         ax1=fig0.add_subplot(figcode)
            #         sw.plot_traces(recording_saved,channel_ids=recording_saved.channel_ids[:32],time_range=[10, 10.07],mode="auto",ax=ax1,add_legend=False)
            #         ax1.title.set_text(f'before whitening')
            #     sw.plot_traces(rec_w,channel_ids=recording_saved.channel_ids[:32],time_range=[10, 10.07],mode="auto",ax=ax,add_legend=False)
            #     ax.title.set_text(f'radius: {this_r}')
            #     i=i+1
            # plt.show()
            # rec_for_sorting = recording_corrected
            
            rec_for_sorting = spre.whiten(
                recording=recording_corrected,
                mode="local",
                radius_um=150,
                #int_scale=200,#this can be added to replicate kilosort behaviour
                dtype=float,
            )
            #sw.plot_traces({f"r100":rec_r100_s200},  mode="auto",time_range=[10, 10.1], backend="ipywidgets")
        ############################# spike sorting ##########################
        #print(f'theses sorters are installed in this PC {ss.installed_sorters()}')
        print(f"run spike sorting with {this_sorter}")
        sorter_params = ss.get_default_sorter_params(this_sorter)
        #print(ss.get_sorter_params_description(this_sorter))
        print(f"the default parameters are: {sorter_params}")
        if this_sorter.startswith("kilosort"):
            #update parameters based on motion correction method
            if motion_corrector =='kilosort_default':
                if probe_type=='H5':
                    sorter_params.update({"nblocks": 1})
                else:
                    sorter_params.update({"nblocks": 0})
                pass
            else:
                if this_sorter == "kilosort3":
                    sorter_params.update({"skip_kilosort_preprocessing": True,"car": False,"do_correction":False})
                else:
                    #sorter_params.update({"skip_kilosort_preprocessing": True,"nblocks": 0})
                    sorter_params.update({"skip_kilosort_preprocessing": True,"do_CAR": False,"do_correction":False,"nblocks": 0})
            #update parameters based on the version of kilosort and probe types
            if this_sorter == "kilosort3":
                ## this will limit the analysis to one PC, try to use kilosort3 with Container in the future.
                kilosort_3_path = r"C:\Users\neuroPC\Documents\GitHub\Kilosort-3.0.2"
                ss.Kilosort3Sorter.set_kilosort3_path(kilosort_3_path)
            else:
                print("use kilosort4")  
                if probe_type=='H10_stacked':
                    sorter_params.update({"dminx": 18.5,"batch_size": 180000})
                elif probe_type=='P2':
                    sorter_params.update({"dminx": 22.5,"nearest_templates": 16, "max_channel_distance": 32,"batch_size": 180000})
                elif probe_type=='H5':
                    sorter_params.update({"dminx": 22.5,"batch_size": 180000})

            if len(recording_corrected_dict)>1:
                rec_for_sorting=si.aggregate_channels(rec_for_sorting)
                sorting_spikes = ss.run_sorter_by_property(
                sorter_name=this_sorter,
                recording=rec_for_sorting,
                grouping_property='group',
                folder=oe_folder / result_folder_name,
                verbose=True,
                **sorter_params
                )
                #engine="joblib",engine_kwargs={"n_jobs": 4}) using this may speed up the analysis
            else:
                sorting_spikes = ss.run_sorter(
                    sorter_name=this_sorter,
                    recording=rec_for_sorting,
                    remove_existing_folder=True,
                    output_folder=oe_folder / result_folder_name,
                    verbose=True,
                    **sorter_params,
                )
        else:
            ### add some lines here to update the parameters based on the sorter type
            #e.g. sorter_params.update({"projection_threshold": [9, 9]})
            sorting_spikes = ss.run_sorter(
                sorter_name=this_sorter,
                recording=rec_for_sorting,
                remove_existing_folder=True,
                output_folder=oe_folder / result_folder_name,
                verbose=True,
                job_kwargs=job_kwargs,
                sorter_params=sorter_params,
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
    # thisDir = r"D:\Open Ephys\2025-03-10_20-25-05"
    #thisDir = r"D:\Open Ephys\2025-03-19_18-02-13"
    #thisDir = r"Z:\DATA\experiment_openEphys\H-series-128channels\2025-03-23_20-47-26"
    #thisDir = r"Z:\DATA\experiment_openEphys\H-series-128channels\2025-03-23_21-33-38"
    #thisDir = r"D:\Open Ephys\2025-04-03_19-13-57"
    #thisDir = r"D:\Open Ephys\2025-04-03_20-36-55"
    thisDir = r"D:\Open Ephys\2025-03-05_13-45-15"
    #thisDir = r"D:\Open Ephys\2025-02-23_20-39-04"
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    raw2si(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
