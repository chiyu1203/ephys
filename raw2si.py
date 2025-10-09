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

global_job_kwargs = dict(n_jobs=n_jobs, chunk_duration="5s", progress_bar=False)
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
    motion_corrector=analysis_methods.get("motion_corrector")
    probe_type = analysis_methods.get("probe_type")
    if probe_type=="P2":
        (win_step_um,win_scale_um)=(30,100)
    else:
        (win_step_um,win_scale_um)=(75,150)
    recording_corrected_dict = {}
    #create a temporary boolean here to account for correct motion not ready to accept dict. If the recording is an Object, it wwill first split it based groups. If an recording Object has no the group attribute,
    #that means it does not go through this line raw_rec = raw_rec.set_probe(probe,group_mode='by_shank') to create the attribute. In this case, a fake Group0 is created just because the function needs that
    #After correcting motion, a dictionary will be created, which can be used in the following analysis
    if type(recording_saved) == dict:
        for group, sub_recording in recording_saved.items():
            print(f"this probe has number of channels to analyse: {len(sub_recording.ids_to_indices())}")
            recording_corrected,motion_info_list=AP_band_drift_estimation(group,sub_recording,oe_folder,analysis_methods,win_step_um,win_scale_um)
            recording_corrected_dict[group]=recording_corrected
    elif len(np.unique(recording_saved.get_property('group')))>1:
        recording_saved = recording_saved.split_by(property='group', outputs='dict')
        for group, sub_recording in recording_saved.items():
            print(f"this probe has number of channels to analyse: {len(sub_recording.ids_to_indices())}")
            recording_corrected,_=AP_band_drift_estimation(group,sub_recording,oe_folder,analysis_methods,win_step_um,win_scale_um)
            recording_corrected_dict[group]=recording_corrected
    else:
        group=0
        recording_corrected,motion_info_list=AP_band_drift_estimation(group,recording_saved,oe_folder,analysis_methods,win_step_um,win_scale_um)
        recording_corrected_dict[group]=recording_corrected
        test_folder = oe_folder /f'motion_shank{group}'/motion_corrector
        if motion_corrector!='testing':
            motion_info=motion_info_list[0]
            motion = motion_info["motion"]
            fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12, 8), sharey=True)
            sw.plot_probe_map(recording_saved, ax=ax1)
            peaks = motion_info["peaks"]
            sr = recording_saved.get_sampling_frequency()
            time_lim0 = 0.0
            time_lim1 = 500.0
            mask = (peaks["sample_index"] > int(sr * time_lim0)) & (peaks["sample_index"] < int(sr * time_lim1))
            sl = slice(None, None, 5)
            amps = np.abs(peaks["amplitude"][mask][sl])
            amps /= np.quantile(amps, 0.95)
            c = plt.get_cmap("inferno")(amps)
            color_kargs = dict(alpha=0.2, s=2, c=c)
            peak_locations = motion_info["peak_locations"]
            ax1.scatter(peak_locations["x"][mask][sl], peak_locations["y"][mask][sl], **color_kargs)
            ax1.set_ylim(-100, 400)
            ax1.set_title('detected peak location')
            peak_locations2 = correct_motion_on_peaks(peaks, peak_locations, motion,recording_saved)
            sw.plot_probe_map(recording_saved, ax=ax2)
            ax2.scatter(peak_locations2["x"][mask][sl], peak_locations2["y"][mask][sl], **color_kargs)
            ax2.set_ylim(-100, 400)
            ax2.set_title('corrected peak location')
            peak_location_figure=test_folder/"corrected_peak_location.png"
            if peak_location_figure.exists() and analysis_methods.get("overwrite_curated_dataset")==False:
                print("the figure exists. analysis methods that does not overwrite it is chosen")
            else:
                fig.savefig(peak_location_figure)
    return recording_corrected_dict

def get_preprocessed_recording(oe_folder,analysis_methods):

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
        tmin_tmax = analysis_methods.get("tmin_tmax")
        this_experimenter = analysis_methods.get("experimenter")
        probe_type = analysis_methods.get("probe_type")
        plot_traces = analysis_methods.get("plot_traces")
        #raw_rec = se.read_openephys(oe_folder, load_sync_timestamps=True)
        raw_rec = se.read_openephys(oe_folder, load_sync_timestamps=True,block_index=0,stream_id='0')
    # To show the start of recording time
    # raw_rec.get_times()[0]
        event = se.read_openephys_event(oe_folder)
        session = Session(oe_folder)
        recording = session.recordnodes[0].recordings[0]
        # camera_trigger_on_oe = recording.events.timestamp[
        #     (recording.events.line == 2) & (recording.events.state == 1)
        # ]
        pd_on_oe = recording.events.timestamp[
            (recording.events.line == 1) & (recording.events.state == 1)
        ]
        pd_off_oe = recording.events.timestamp[
            (recording.events.line == 1) & (recording.events.state == 0)
        ]
        np.save(oe_folder/"pd_on.npy",pd_on_oe)
        np.save(oe_folder/"pd_off.npy",pd_off_oe)
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
        # Slice the recording if needed
        if tmin_tmax[1]>0 and tmin_tmax[1]>tmin_tmax[0]:
            start_sec = tmin_tmax[0]
            end_sec = tmin_tmax[1]
            rec_of_interest = raw_rec.frame_slice(start_frame=start_sec * fs, end_frame=end_sec * fs)
        elif tmin_tmax[1]<0:
            print("tmax <0 means to analyse the entire recording")
        else:
            ValueError("tmax needs to be bigger than tmin to select certain section of the recording")

        ################load probe information################
        if probe_type == "H10_stacked":
            stacked_probes = pi.read_probeinterface("H10_stacked_probes_2D.json")
            probe = stacked_probes.probes[0]
        elif probe_type == "H10_rev":
            probe_name= "ASSY-77-H10"
            stacked_probes = pi.read_probeinterface("H10_RHD2164_rev_openEphys_mapping.json")
            probe = stacked_probes.probes[0]
        elif probe_type == "P2":
            probe_name= "ASSY-37-P-2"
            stacked_probes = pi.read_probeinterface("P2_RHD2132_openEphys_mapping.json")
            probe = stacked_probes.probes[0]    
        else:
            manufacturer = "cambridgeneurotech"
            if probe_type == "H5":
                probe_name = "ASSY-77-H5"
                connector_type="ASSY-77>Adpt.A64-Om32_2x-sm-cambridgeneurotech>RHD2164"
            elif probe_type == "H10":
                probe_name = "ASSY-77-H10"
                connector_type="ASSY-77>Adpt.A64-Om32_2x-sm-cambridgeneurotech>RHD2164"
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
        raw_rec = raw_rec.set_probe(probe,group_mode='by_shank')
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
        elif plot_traces:
            raw_rec_dict = raw_rec.split_by(property='group', outputs='dict')
            fig0=plt.figure(figsize=[64,48])
            for group, rec_per_shank in raw_rec_dict.items():
                figcode=int(f"22{group+1}")
                ax=fig0.add_subplot(figcode)
                sw.plot_traces(rec_per_shank,  mode="auto",ax=ax)
            fig0.savefig(oe_folder /'before_band_pass_and_remove_channels.png')
        ################ preprocessing ################
        # apply band pass filter
        ### need to double check whether there is a need to convert data type to float32. It seems that this will increase the size of the data
        recording_f = spre.bandpass_filter(raw_rec, freq_min=600, freq_max=6000)
        #recording_f = spre.bandpass_filter(raw_rec, freq_min=600, freq_max=6000,filter_order=2,ftype="bessel") ## in the hands-on tutorial, one speaker use bessel filter
        #recording_f = spre.bandpass_filter(raw_rec, freq_min=600, freq_max=6000,dtype="float32")# it sounds that people recommend to run two separate bandpass filter for motion estimation and for spike sorting.
        # recording_f = spre.highpass_filter(raw_rec, freq_min=300,dtype="float32")
        

        if analysis_methods.get("remove_dead_channels")==True:
            """
            This step should be done before saving preprocessed files because ideally the preprocessed file we want to create is something ready for spiking
            detection, which means neural traces gone through bandpass filter and common reference.
            However, applying common reference takes signals from channels of interest which requires us to decide what we want to do with other bad or noisy channels first.
            """
            if probe_type == "H10_rev":
                broken_shank_ids=np.array(['CH33','CH34','CH35','CH36','CH37','CH38','CH39','CH40','CH41','CH42','CH43','CH44','CH45','CH46','CH47','CH48','CH49','CH50','CH51','CH52','CH53','CH54','CH55','CH56','CH57','CH58','CH59','CH60','CH61','CH62','CH63','CH64'])
                recording_f = recording_f.remove_channels(
                        broken_shank_ids
                    )
            elif probe_type == "H10":
                broken_shank_ids=np.array(['CH1','CH2','CH3','CH4','CH5','CH6','CH7','CH8','CH9','CH10','CH11','CH12','CH13','CH14','CH15','CH16','CH17','CH18','CH19','CH20','CH21','CH22','CH23','CH24','CH25','CH26','CH27','CH28','CH29','CH30','CH31','CH32'])
                recording_f = recording_f.remove_channels(
                        broken_shank_ids
                    )
            bad_channel_ids,channel_labels = spre.detect_bad_channels(recording_f)
            #
            # (noise_inds,) = np.where(channel_labels=='noise')
            # noise_channel_ids = recording_f.channel_ids[noise_inds]
            (dead_inds,) = np.where(channel_labels=='dead')
            dead_channel_ids = recording_f.channel_ids[dead_inds]
            print("bad_channel_ids", bad_channel_ids)
            print("channel_labels", channel_labels)
            if analysis_methods.get("analyse_good_channels_only") == True:
                recording_f = recording_f.remove_channels(
                    bad_channel_ids
                )  #try this functino interpolate_bad_channels when I can put 3 shanks in the brain plus when there is some noisy channels
                #https://spikeinterface.readthedocs.io/en/stable/api.html#spikeinterface.preprocessing.interpolate_bad_channels
            elif analysis_methods.get("interpolate_noisy_channels")==True:
                ##this function is still buggy so just treat this option as keeping noisy channels while removing dead channels
                recording_f = recording_f.remove_channels(dead_channel_ids)
                #recording_f=spre.interpolate_bad_channels(dead_channel_ids)
            else: 
                pass
        ##start to split the recording into groups here because remove bad channels function is not ready to receive dict as input
        recordings_dict = recording_f.split_by(property='group', outputs='dict')

        fig0=plt.figure(figsize=[64,48])
        for group, rec_per_shank in recordings_dict.items():
            figcode=int(f"22{group+1}")
            ax=fig0.add_subplot(figcode)
            sw.plot_traces(rec_per_shank,  mode="auto",ax=ax)
        fig0.savefig(oe_folder /'after_band_pass_and_remove_channels.png')
        if plot_traces:
            fig0.show()
            #shankid=0
            #sw.plot_traces({f"shank{shankid+1}": recordings_dict[shankid]},  mode="auto",time_range=[10, 10.1], backend="ipywidgets")
            #sw.plot_traces(recordings_dict[shankid],  mode="auto",time_range=[10, 10.1])
        

        # apply common median reference to remove common noise
        recording_cmr = spre.common_reference(
            recordings_dict, reference="global", operator="median"
        )
        fig1=plt.figure(figsize=[64,48])
        for group, rec_per_shank in recording_cmr.items():
            figcode=int(f"12{group+1}")
            ax=fig1.add_subplot(figcode)
            sw.plot_traces(rec_per_shank,  mode="auto",ax=ax)
        fig1.savefig(oe_folder /'after_cmr.png')
        # recording_cmr = spre.common_reference(
        #     recording_f, reference="global", operator="median" 
        # )
        '''
        ref_channel_idslist | str | int | None, default: None
If “global” reference, a list of channels to be used as reference. If “single” reference, a list of one channel or a single channel id is expected. If “groups” is provided, then a list of channels to be applied to each group is expected.
        '''

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
        # recordings_dict = rec_of_interest.split_by(property='group', outputs='dict')
        # fig0=plt.figure()
        # for group, rec_per_shank in recordings_dict.items():
        #     figcode=int(f"22{group+1}")
        #     ax=fig0.add_subplot(figcode)
        #     sw.plot_traces(rec_per_shank,  mode="auto",ax=ax)
        # plt.show()
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
    
    if (oe_folder / sorting_folder_name).is_dir() and analysis_methods.get("overwrite_curated_dataset") == False:
        sorting_spikes = si.load_extractor(oe_folder / sorting_folder_name)
        w_rs = sw.plot_rasters(sorting_spikes, time_range=(0, 30), backend="matplotlib")
        return print("this data is processed already.")
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
                        cname=compressor_name, clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE
                    )
                    if type(rec_of_interest) == dict:#create a temporary boolean here to account for that save.() function can not take dict as input
                        rec_of_interest=si.aggregate_channels(rec_of_interest)
                    recording_saved = rec_of_interest.save(
                        format="zarr",
                        folder=oe_folder / "preprocessed_compressed.zarr",
                        compressor=compressor,
                        overwrite=True,
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
        recording_saved=spre.astype(recording_saved,np.float32)
        recording_corrected_dict=motion_correction_shankbyshank(recording_saved,oe_folder,analysis_methods)

        # if plot_traces:
        #     fig1=plt.figure()
        #     for group, rec_per_shank in recording_corrected_dict.items():
        #         figcode=int(f"22{group+1}")
        #         ax=fig1.add_subplot(figcode)
        #         sw.plot_traces(rec_per_shank,  mode="auto",ax=ax)
        #     plt.show()   

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
            # r_range=[25,50,100,150]
            # #recording_saved.channel_ids[:10]np.linspace(1,10,10,dtype=int)
            # i=0
            # for this_r in r_range:
            #     rec_w = spre.whiten(recording=recording_saved,mode="local",radius_um=this_r,int_scale=200,dtype=float)
            #     figcode=int(f"15{i+1}")
            #     ax=fig0.add_subplot(figcode)
            #     if i==3:
            #         figcode=int(f"15{i+2}")
            #         ax1=fig0.add_subplot(figcode)
            #         sw.plot_traces(recording_saved,channel_ids=recording_saved.channel_ids[:32],time_range=[500, 500.1],mode="auto",ax=ax1,add_legend=False)
            #         ax1.title.set_text(f'before whitening')
            #     sw.plot_traces(rec_w,channel_ids=recording_saved.channel_ids[:32],time_range=[500, 500.1],mode="auto",ax=ax,add_legend=False)
            #     ax.title.set_text(f'radius: {this_r}')
            #     i=i+1
            # plt.show()
            # rec_for_sorting = recording_corrected
            
            rec_for_sorting = spre.whiten(
                recording=recording_corrected,
                mode="local",
                radius_um=150,
                #int_scale=200,#this can be added to replicate kilosort behaviour
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
                if probe_type.startswith('H10') :
                    #sorter_params.update({"dminx": 18.5,"batch_size": 60000,"nearest_templates": 16})#change the batch size here due to an error according to  https://github.com/MouseLand/Kilosort/issues/719 The error probably comes from after removing some channels manually
                    sorter_params.update({"dminx": 18.5,"batch_size": 180000,"nearest_templates": 32})
                elif probe_type=='P2':
                    sorter_params.update({"dminx": 22.5,"batch_size": 180000,"nearest_templates": 16})
                elif probe_type=='H5':
                    sorter_params.update({"dminx": 22.5,"batch_size": 180000})

            if len(recording_corrected_dict)>1:
                rec_for_sorting=si.aggregate_channels(rec_for_sorting)
                sorting_spikes = ss.run_sorter_by_property(
                sorter_name=this_sorter,
                recording=rec_for_sorting,
                remove_existing_folder=True,
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
                    folder=oe_folder / result_folder_name,
                    verbose=True,
                    **sorter_params,
                )
        else:
            ### add some lines here to update the parameters based on the sorter type
            #e.g. sorter_params.update({"projection_threshold": [9, 9]})
            sorter_params.update({"apply_motion_correction": False,"apply_preprocessing": False})
            #sorter_params['general'].update({"radius_um":150})
            sorter_params['cache_preprocessing'].update({"mode": "no-cache"})
            if len(recording_corrected_dict)>1:#it sounds like skipping motion correction will lead here with only one dict. Note: correct motion correction will remove channels
                rec_for_sorting=si.aggregate_channels(rec_for_sorting)
                sorting_spikes = ss.run_sorter_by_property(
                sorter_name=this_sorter,
                recording=rec_for_sorting,
                remove_existing_folder=True,
                grouping_property='group',
                folder=oe_folder / result_folder_name,
                verbose=True,
                **sorter_params
                )
            else:

                sorting_spikes = ss.run_sorter(
                    sorter_name=this_sorter,
                    recording=rec_for_sorting,
                    remove_existing_folder=True,
                    folder=oe_folder / result_folder_name,
                    verbose=True,**sorter_params,
                )
        ##this will return a sorting object
    ############################# spike sorting preview and saving ##########################
        w_rs = sw.plot_rasters(sorting_spikes, time_range=(0, 30), backend="matplotlib")
        if (
            analysis_methods.get("save_sorting_file") == True
            and analysis_methods.get("overwrite_curated_dataset") == True
        ):
            sorting_spikes.save(folder=oe_folder / sorting_folder_name, overwrite=True)
            json_string = json.dumps(analysis_methods, indent=1)
            with open(oe_folder / sorting_folder_name / "analysis_methods_dictionary_backup.json", "w") as f:
                f.write(json_string)

    return print("Spiking sorting done. The rest of the tasks can be done in other PCs")


if __name__ == "__main__":
    #thisDir = r'Y:/GN25031/250803/looming/session2/2025-08-03_21-24-13'#bad_channel_ids ['CH3' 'CH5' 'CH6' 'CH7' 'CH10']
    #thisDir = r'Y:/GN25034/250907/gratings/session1/2025-09-08_00-40-55'#bad_channel_ids ['CH5' 'CH7' 'CH10']
    #thisDir = r'Y:/GN25034/250907/coherence/session1/2025-09-08_01-12-18'#bad_channel_ids ['CH24']
    #thisDir = r'Y:\GN25034\250907\gratings\session1\2025-09-08_00-40-55'#bad_channel_ids ['CH24']
    #thisDir = r'Y:\GN25038\250924\gratings\session1\2025-09-24_18-40-05'#bad_channel_ids ['CH3']
    thisDir = r"Y:\GN25029\250729\looming\session1\2025-07-29_15-22-54"#
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    raw2si(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
