import time, os, json, warnings, sys
import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.qualitymetrics as sqm
import spikeinterface.widgets as sw
import spikeinterface.curation as sc
from datetime import datetime
from scipy.signal import medfilt,convolve, gaussian
# use elepant or open scope to faciliate data analysis https://elephant.readthedocs.io/en/latest/index.html
# For kilosort/phy output files we can use the read_phy
# most formats will have a read_xx that can used.
import matplotlib.pyplot as plt
from spikeinterface.widgets import plot_sorting_summary
import numpy as np
from pathlib import Path
import pandas as pd
import pynapple as nap
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.plot import peri_event_time_histogram, driftmap_color, driftmap
from brainbox.ephys_plots import (
    plot_cdf,
    image_rms_plot,
    scatter_raster_plot,
    scatter_amp_depth_fr_plot,
    probe_rms_plot,
)
from spike_curation import (
    calculate_analyzer_extension,
    get_preprocessed_recording,
    spike_overview,
    generate_sorter_suffix,
    MplColorHelper,
)
warnings.simplefilter("ignore")
current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(
    0, str(parent_dir) + "\\utilities"
)  ## 0 means search for new dir first and 1 means search for sys.path first
from useful_tools import find_file
from data_cleaning import sorting_trial_info, load_fictrac_data_file,euclidean_distance

sys.path.insert(0, str(parent_dir) + "\\bonfic")
from analyse_stimulus_evoked_response import classify_trial_type,preprocess_tracking_data,identify_behavioural_states,generate_index_points
n_cpus = os.cpu_count()
n_jobs = n_cpus - 4
global_job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)
# global_job_kwargs = dict(n_jobs=16, chunk_duration="5s", progress_bar=False)
si.set_global_job_kwargs(**global_job_kwargs)

def string2datetime(date_time):
    #format = 'YYYY-MM-DD_HH_MM_SS'
    format = "%Y-%m-%d_%H-%M-%S"
    datetime_str = datetime.strptime(date_time, format)
    
    return datetime_str
# based on chatGPT
def sort_arrays(arr1, *arrays):
    # Combine all arrays into tuples
    combined = zip(arr1, *arrays)

    # Sort based on the first array
    sorted_combined = sorted(combined, key=lambda x: x[0])

    # Extract the sorted arrays
    sorted_arr1 = [pair[0] for pair in sorted_combined]
    sorted_arrays = [
        [pair[i] for pair in sorted_combined] for i in range(1, len(arrays) + 1)
    ]

    return sorted_arr1, *sorted_arrays

def calculate_peths_details(
    spike_times, spike_clusters, cluster_ids, align_times, pre_time=0.2,
    post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True):
    ## this is a function from calculate_peths singlecell.py in brainbox library
    n_offset = 5 * int(np.ceil(smoothing / bin_size))  # get rid of boundary effects for smoothing
    n_bins_pre = int(np.ceil(pre_time / bin_size)) + n_offset
    n_bins_post = int(np.ceil(post_time / bin_size)) + n_offset
    n_bins = n_bins_pre + n_bins_post
    binned_spikes = np.zeros(shape=(len(align_times), len(cluster_ids), n_bins))

    # build gaussian kernel if requested
    if smoothing > 0:
        w = n_bins - 1 if n_bins % 2 == 0 else n_bins
        window = gaussian(w, std=smoothing / bin_size)
        # half (causal) gaussian filter
        # window[int(np.ceil(w/2)):] = 0
        window /= np.sum(window)
        binned_spikes_conv = np.copy(binned_spikes)

    ids = np.unique(cluster_ids)
    #ids = cluster_ids

    # # filter spikes outside of the loop
    idxs = np.bitwise_and(spike_times >= np.min(align_times) - (n_bins_pre + 1) * bin_size,
                          spike_times <= np.max(align_times) + (n_bins_post + 1) * bin_size)
    idxs = np.bitwise_and(idxs, np.isin(spike_clusters, cluster_ids))
    spike_times = spike_times[idxs]
    spike_clusters = spike_clusters[idxs]

    # compute floating tscale
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    # bin spikes
    for i, t_0 in enumerate(align_times):
        # define bin edges
        ts = tscale + t_0
        # filter spikes
        idxs = np.bitwise_and(spike_times >= ts[0], spike_times <= ts[-1])
        i_spikes = spike_times[idxs]
        i_clusters = spike_clusters[idxs]

        # bin spikes similar to bincount2D: x = spike times, y = spike clusters
        xscale = ts
        xind = (np.floor((i_spikes - np.min(ts)) / bin_size)).astype(np.int64)
        yscale, yind = np.unique(i_clusters, return_inverse=True)
        nx, ny = [xscale.size, yscale.size]
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
        r = np.bincount(ind2d, minlength=nx * ny, weights=None).reshape(ny, nx)

        # store (ts represent bin edges, so there are one fewer bins)
        bs_idxs = np.isin(ids, yscale)
        binned_spikes[i, bs_idxs, :] = r[:, :-1]

        # smooth
        if smoothing > 0:
            idxs = np.where(bs_idxs)[0]
            for j in range(r.shape[0]):
                binned_spikes_conv[i, idxs[j], :] = convolve(
                    r[j, :], window, mode='same', method='auto')[:-1]
    # average
    if smoothing > 0:
        binned_spikes_ = np.copy(binned_spikes_conv)
    else:
        binned_spikes_ = np.copy(binned_spikes)
    if return_fr:
        binned_spikes_ /= bin_size

    peth_means = np.mean(binned_spikes_, axis=0)
    peth_stds = np.std(binned_spikes_, axis=0)

    if smoothing > 0:
        peth_means = peth_means[:, n_offset:-n_offset]
        peth_stds = peth_stds[:, n_offset:-n_offset]
        binned_spikes = binned_spikes[:, :, n_offset:-n_offset]
        tscale = tscale[n_offset:-n_offset]

    # package output
    tscale = (tscale[:-1] + tscale[1:]) / 2
    #peths = Bunch({'means': peth_means, 'stds': peth_stds, 'tscale': tscale, 'cscale': ids})
    return peth_means, peth_stds, tscale,ids



def align_async_signals(oe_folder, json_file):
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    ## load previous analysis methods
    stim_variable2 = analysis_methods.get("stim_variable2",'Duration')
    load_previous_methods=analysis_methods.get("load_previous_methods",False)
    if load_previous_methods:
        previous_methods_file=find_file(oe_folder, "ephys_analysis_methods_backup.json")
        if previous_methods_file!=None:
            with open(previous_methods_file, "r") as f:
                print(f"load analysis methods from previous file {previous_methods_file}")
                previous_analysis_methods = json.loads(f.read())
            analysis_methods.update(previous_analysis_methods)
        else:
            print("previous analysis methods file is not found. Use the current one.")

    if type(oe_folder)==str:
        oe_folder = Path(oe_folder)        
    exp_datetime = string2datetime(oe_folder.stem)
    this_sorter = analysis_methods.get("sorter_name")
    this_experimenter = analysis_methods.get("experimenter")
    experiment_name = analysis_methods.get("experiment_name")
    if experiment_name in ['gratings','coherence']:
        stationary_phase_before_motion=False
    else:
        stationary_phase_before_motion = analysis_methods.get("stationary_phase_before_motion",True)
    sorter_suffix = generate_sorter_suffix(this_sorter)
    result_folder_name = "results" + sorter_suffix
    sorting_folder_name = "sorting" + sorter_suffix
    analyser_folder_name = "analyser" + sorter_suffix
    phy_folder_name = phy_folder_name = "phy" + sorter_suffix
    report_folder_name = "report" + sorter_suffix
    
    #looking for files in oe folder
    pd_ext='pd_*.npy'
    pd_files = find_file(oe_folder, pd_ext)
    pd_ext='pd.npy'
    one_pd_file = find_file(oe_folder, pd_ext)
    camera_ext='camera_*.npy'
    camera_sync_file = find_file(oe_folder, camera_ext)
    ### needs to figure out whether this section is necessary or not
    # if pd_files is not None and analyse_spontaneous_activity is False:
    #     pd_on_oe=np.load(pd_files[1])
    #     pd_off_oe=np.load(pd_files[0])
    # elif one_pd_file is not None and analyse_spontaneous_activity is False:
    #     pd_on_oe=np.load(one_pd_file)[0]
    #     pd_off_oe=np.load(one_pd_file)[1]
    # elif analyse_spontaneous_activity is False:
    # ##load adc events from openEphys
    #     event = se.read_openephys_event(oe_folder)
    #     evts=event.get_events(channel_id=event.channel_ids[0])
    #     pd_data=evts[evts['label']=='1']
    #     camera_data=evts[evts['label']=='2']
    #     barcode_data=evts[evts['label']=='3']
    #     if pd_data.shape[0]>1:
    #         pd_on=pd_data['time']
    #         pd_off=pd_data['time']+pd_data['duration']
    #         np.save(oe_folder/"pd.npy",np.vstack((pd_on,pd_off)))
    #     if camera_data.shape[0]>1:
    #         np.save(oe_folder/"camera_pulse.npy",camera_data['time'])
    #     if barcode_data.shape[0]>1:
    #         barcode_on=barcode_data['time']
    #         barcode_off=barcode_data['time']+barcode_data['duration']
    #         np.save(oe_folder/"barcode.npy",np.vstack((barcode_on,barcode_off)))
    # elif analyse_spontaneous_activity is True and camera_sync_file is None:
    #     print('need to organise the code here to load ephys event data')
    
    ## looking for files in the previous folder
    stim_directory = oe_folder.resolve().parents[0]
    database_ext = "database*"
    raw_tracking = find_file(stim_directory, database_ext)
    video_ext = "*.mp4"
    video_file = find_file(stim_directory, video_ext)
    if raw_tracking is None and video_file is not None:
        print("fictrac database file is not found yet. Either the fictrac analysis is not done or not converted to .parquet file")
        print("however, it is better to analyse fictrac data first")
        fictrac_data=load_fictrac_data_file(video_file, analysis_methods,column_to_drop=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23])
    # stim_sync_ext = "*_stim_sync.csv"
    # stim_sync_file = find_file(stim_directory, stim_sync_ext)
    velocity_ext = "velocity_tbt.npy"
    velocity_file = find_file(stim_directory, velocity_ext)
    rotation_ext = "angular_velocity_tbt.npy"
    rotation_file = find_file(stim_directory, rotation_ext)
    trial_ext = "trial*.csv"
    trial_file = find_file(stim_directory, trial_ext)
    event_of_interest = analysis_methods.get("event_of_interest")
    stim_duration = analysis_methods.get("stim_duration")
    ISI_duration = analysis_methods.get("interval_duration")
    preStim_duration =analysis_methods.get("prestim_duration")
    time_window = analysis_methods.get("analysis_window")#### there should be a analysis window to do data analysis
    #behavioural related metrics
    analyse_behavioural_state_modulation =analysis_methods.get("analyse_behavioural_state_modulation",False)
    camera_fps = analysis_methods.get("camera_fps")
    filtering_method=analysis_methods.get("filtering_method")
    yaw_axis=analysis_methods.get("yaw_axis")
    smooth_window_length = round(0.5*camera_fps)
    smooth_window_length = smooth_window_length if np.mod(smooth_window_length, 2) == 1 else smooth_window_length + 1
    #plotting related metrics
    colormap_name = "coolwarm"
    COL = MplColorHelper(colormap_name, 0, 8)

    ### in genereral, there are 4 scenarios in the ephys
    ### 1st, spontaneous activity with video (when the video is not avaiable, there is nothing to decode), no need to load raw trial info
    ### 2nd, visual evoked activity with video needs to interpolate behavioural time stamp with ephys timestamp, no need to load raw trial info
    ### 3rd, visual evoked activity with video sync needs to align behavioural time stamp with ephys timestamp, no need to load raw trial info
    ### 4th, visual evoked activity without video (visual only) needs load raw trial info

    #time_window = np.array([-5, 5])
    if trial_file is None:
        print("analyse spontaneous activity")
        if camera_sync_file is not None:
            oe_camera_time=np.load(camera_sync_file)
            tracking_df=pd.read_parquet(raw_tracking)
        else:
            print("no stimlus timestamp and no camera timestamp. Unable to do further analysis")
            return
    else:
        if video_file is None:
            print("analyse ephys data only. Load raw trial information from csv file")
            stim_pd = pd.read_csv(trial_file)
            meta_info, stim_type = sorting_trial_info(stim_pd,analysis_methods)
            if experiment_name=='coherence':#RDK is special because isi info is also logged in the trial info
                meta_info = meta_info[1::2]
        else:
            summary_ext = "behavioural_summary.parquet.gzip"
            stimulus_meta_file = find_file(stim_directory, summary_ext)
            if stimulus_meta_file is None:
                if trial_file is None:
                    print("behavioural summary file and raw trial file not found")
                    return
                stim_pd = pd.read_csv(trial_file)
                meta_info, stim_type = sorting_trial_info(stim_pd,analysis_methods)
                if experiment_name=='coherence':#RDK is special because isi info is also logged in the trial info
                    meta_info = meta_info[1::2]
            else:
                meta_info = pd.read_parquet(stimulus_meta_file)
            if 'present_trial_duration' in meta_info.columns:
                meta_info['Duration']=meta_info['present_trial_duration']#create a new column called Duration temporary to procastinate unifying the variable name
        
        ### changed ISI and Stim signals to 1 and 0 from 2025 April 1st        
        stim_type=meta_info['stim_type'].unique()
        num_stim = meta_info.shape[0]
        if experiment_name in ['looming',"receding","conflict","sweeping","flashing"]:
            if 'PreMovDuration' in meta_info.columns:
                if meta_info['PreMovDuration'].unique()==0:
                    pd_on_oe=pd_on_oe[preStim_duration<pd_on_oe]
                    pd_off_oe=pd_off_oe[preStim_duration<pd_off_oe]
                    if pd_off_oe[0]>pd_on_oe[0]:
                        stim_on_oe = pd_on_oe[:num_stim]
                        isi_on_oe = pd_off_oe[:num_stim]
                    else:
                        stim_on_oe = pd_off_oe[:num_stim]
                        isi_on_oe = pd_on_oe[:num_stim]
                else:
                    pd_on_oe=pd_on_oe[preStim_duration<pd_on_oe]
                    pd_off_oe=pd_off_oe[preStim_duration<pd_off_oe]
                    if 'gregarious_locust' in stim_type and exp_datetime < datetime(2025, 11, 1, 12, 0, 0):
                        pd_on_oe=pd_on_oe[1:]
                        pd_off_oe=pd_off_oe[1:]
                    if pd_off_oe[0]<pd_on_oe[0] and pd_on_oe[0]-pd_off_oe[0]<0.8: #for some reason in GN25048, pd_off_oe happens before pd_on_oe
                        stim_on_oe = pd_on_oe[::2]## if bright stimuli represent appearance of the stimulus or the stop of moving stimulus, then stim onset and ISI onset are based on pd_off_oe
                        isi_on_oe=pd_on_oe[1::2]            
                    else:
                        stim_on_oe = pd_off_oe[::2]## if bright stimuli represent appearance of the stimulus or the stop of moving stimulus, then stim onset and ISI onset are based on pd_off_oe
                        isi_on_oe=pd_off_oe[1::2]            
            elif stationary_phase_before_motion==True:## this is used in gratings or if the meta info was first processed by bonfic code where the 'PreMovDuration' is not in the meta_info.columns.
                pd_on_oe=pd_on_oe[preStim_duration<pd_on_oe]
                pd_off_oe=pd_off_oe[preStim_duration<pd_off_oe]
                if 'gregarious_locust' in stim_type and exp_datetime < datetime(2025, 11, 1, 12, 0, 0):##the date before the bug in locust loom is fixed
                    pd_on_oe=pd_on_oe[1:]
                    pd_off_oe=pd_off_oe[1:]
                if pd_off_oe[0]<pd_on_oe[0] and pd_on_oe[0]-pd_off_oe[0]<0.8: #for some reason in GN25048, pd_off_oe happens before pd_on_oe
                    stim_on_oe = pd_on_oe[::2]## if bright stimuli represent appearance of the stimulus or the stop of moving stimulus, then stim onset and ISI onset are based on pd_off_oe
                    isi_on_oe=pd_on_oe[1::2]            
                else:
                    stim_on_oe = pd_off_oe[::2]## if bright stimuli represent appearance of the stimulus or the stop of moving stimulus, then stim onset and ISI onset are based on pd_off_oe
                    isi_on_oe=pd_off_oe[1::2]
            else:
                pd_on_oe=pd_on_oe[preStim_duration<pd_on_oe]
                pd_off_oe=pd_off_oe[preStim_duration<pd_off_oe]
                if pd_off_oe[0]>pd_on_oe[0]:
                    stim_on_oe = pd_on_oe[:num_stim]
                    isi_on_oe = pd_off_oe[:num_stim]
                else:
                    stim_on_oe = pd_off_oe[:num_stim]
                    isi_on_oe = pd_on_oe[:num_stim]
        else:
            stim_on_oe = pd_on_oe
            isi_on_oe = pd_off_oe
        if camera_sync_file is not None and video_file is not None:
            oe_camera_time=np.load(camera_sync_file)
            tracking_df=pd.read_parquet(raw_tracking)
        elif video_file is not None:
            print("do interpolation across each timestamp")

    if 'tracking_df' in locals():
        x_all,y_all,yaw_angular_velocity=preprocess_tracking_data(tracking_df,filtering_method,yaw_axis,smooth_window_length)
        travel_distance_fbf = euclidean_distance(x_all,y_all)
        walk_states,turn_states,turn_cw,turn_ccw,fig=identify_behavioural_states(travel_distance_fbf,yaw_angular_velocity,filtering_method,camera_fps,consecutive_duration=[1,0.5],smooth_window_length=smooth_window_length,skip_smoothing=False)
        plot_name = f"behavioural_states_classification.png"
        fig.savefig(oe_folder / plot_name)
        if walk_states[0]==1:#identify transition onset depends on whether the first frame is run or stationary already
            s2w_index=np.where(np.diff(walk_states))[0][1::2]
            w2s_index=np.where(np.diff(walk_states))[0][::2]
        else:
            s2w_index=np.where(np.diff(walk_states))[0][::2]
            w2s_index=np.where(np.diff(walk_states))[0][1::2]
        w2s_index=w2s_index[w2s_index+time_window[1]*camera_fps<travel_distance_fbf.shape[0]]#remove events passing the end of recording
        s2w_index=s2w_index[s2w_index+time_window[1]*camera_fps<travel_distance_fbf.shape[0]]
        w2s_index=w2s_index[w2s_index-abs(time_window[0]*camera_fps)>0]#remove events starting before the start of recording
        s2w_index=s2w_index[s2w_index-abs(time_window[0]*camera_fps)>0]
        if turn_states[0]==1:
            n2t_index=np.where(np.diff(turn_states))[0][1::2]
        else:
            n2t_index=np.where(np.diff(turn_states))[0][::2]
        n2t_index=n2t_index[n2t_index+time_window[1]*camera_fps<travel_distance_fbf.shape[0]]
        n2t_index=n2t_index[n2t_index-abs(time_window[0]*camera_fps)>0]

    if event_of_interest.lower() == "preStim_ISI":
        events_time = isi_on_oe[1:].values
    elif event_of_interest.lower() == "postStim_ISI":
        events_time = isi_on_oe[:-1].values
    elif event_of_interest.lower() == "walk_onset":
        events_time = oe_camera_time[s2w_index]
        transition_index=s2w_index
    elif event_of_interest.lower() == "stop_onset":
        events_time = oe_camera_time[w2s_index]
        transition_index=w2s_index
    elif event_of_interest.lower() == "turn_onset":
        events_time = oe_camera_time[n2t_index]
        transition_index=n2t_index
    elif event_of_interest.lower() == "walk_straight_onset":
        walk_straight_index=np.where((turn_states[1:] == 0) & (walk_states == 1))[0]
        walk_straight_states=np.zeros(len(walk_states))
        walk_straight_states[walk_straight_index]=1
        if walk_straight_states[0]==1:
            s2w_index=np.where(np.diff(walk_states))[0][1::2]
        else:
            s2w_index=np.where(np.diff(walk_straight_states))[0][::2]
        s2w_index=s2w_index[s2w_index+time_window[1]*camera_fps<travel_distance_fbf.shape[0]]
        s2w_index=s2w_index[s2w_index-abs(time_window[0]*camera_fps)>0]
        events_time = oe_camera_time[s2w_index]
        transition_index=s2w_index
    elif event_of_interest.lower() == "turn_ccw_onset":
        turn_index=np.where((turn_ccw[1:] == 1) & (walk_states == 1))[0]
        turn_states=np.zeros(len(walk_states))
        turn_states[turn_index]=1
        if turn_states[0]==1:
            n2t_index=np.where(np.diff(turn_states))[0][1::2]
        else:
            n2t_index=np.where(np.diff(turn_states))[0][::2]
        n2t_index=n2t_index[n2t_index+time_window[1]*camera_fps<travel_distance_fbf.shape[0]]
        n2t_index=n2t_index[n2t_index-abs(time_window[0]*camera_fps)>0]
        events_time = oe_camera_time[n2t_index]
        transition_index=n2t_index
    elif event_of_interest.lower() == "turn_cw_onset":
        turn_index=np.where((turn_cw[1:] == 1) & (walk_states == 1))[0]
        turn_states=np.zeros(len(walk_states))
        turn_states[turn_index]=1
        if turn_states[0]==1:
            n2t_index=np.where(np.diff(turn_states))[0][1::2]
        else:
            n2t_index=np.where(np.diff(turn_states))[0][::2]
        n2t_index=n2t_index[n2t_index+time_window[1]*camera_fps<travel_distance_fbf.shape[0]]
        n2t_index=n2t_index[n2t_index-abs(time_window[0]*camera_fps)>0]
        events_time = oe_camera_time[n2t_index]
        transition_index=n2t_index
    else:
        if len(stim_on_oe) > num_stim:
            stim_on_oe=stim_on_oe[preStim_duration<stim_on_oe]### I should move this to line 258 for coherence etc.
            events_time=stim_on_oe[:num_stim]
        elif type(stim_on_oe)==np.ndarray:
            events_time = stim_on_oe
        else:
            events_time = stim_on_oe[:].values

    ##build up analysis time window
    events_time_tw = np.array(
        [events_time + time_window[0], events_time + time_window[1]]
    ).T
    if 'transition_index' in locals():
        if transition_index.shape[0]>0:
            index_points = generate_index_points(transition_index, time_window, camera_fps)
            # if event_of_interest.lower().startswith('walk') or event_of_interest.lower().startswith('stop'):      
            time_locked_behaviour=travel_distance_fbf[index_points]*camera_fps
            # elif event_of_interest.lower().startswith('turn'):
            time_locked_behaviour2=yaw_angular_velocity[index_points]
        else:
            print('no transition detected')
            return
        time_points = np.arange(time_locked_behaviour.shape[1])
        time_points_b = np.tile(time_points, (time_locked_behaviour.shape[0], 1))
        fig, (axes,axes2) = plt.subplots(
            nrows=2, ncols=1, figsize=(9, 4.5), tight_layout=True,sharex=True
        )#change fig size to 11,5.5 or something lower value for bigger legend
        mean=np.mean(time_locked_behaviour, axis=0)
        frame_index=np.mean(time_points_b, axis=0)
        bars=np.std(time_locked_behaviour, axis=0)
        axes.plot(
            frame_index,
            mean,
            color='red',
            linewidth=4
        )
        axes.plot(
            np.transpose(time_points_b),
            np.transpose(time_locked_behaviour),
            color='black',
            linewidth=0.2,
            alpha=0.5
        )
        axes.set_xticks([0,abs(time_window[0])*camera_fps,(time_window[1]+abs(time_window[0]))*camera_fps])
        axes.set_xticklabels([time_window[0],0,time_window[1]])
        axes.set_ylabel('Speed (cm/sec)',size=15)#size should be above 10
        axes.spines['left'].set_linewidth(2) 
        time_points = np.arange(time_locked_behaviour2.shape[1])
        time_points_b = np.tile(time_points, (time_locked_behaviour2.shape[0], 1))
        median=np.median(time_locked_behaviour2, axis=0)
        frame_index=np.mean(time_points_b, axis=0)
        bars=np.std(time_locked_behaviour2, axis=0)
        axes2.plot(
            frame_index,
            median,
            color='red',
            linewidth=4
        )
        axes2.plot(
            np.transpose(time_points_b),
            np.transpose(time_locked_behaviour2),
            color='black',
            linewidth=0.2,
            alpha=0.5
        ) 
        axes2.set_ylabel(r'$\omega$ (rad/sec)',size=15)
        axes2.set_xticks([0,abs(time_window[0])*camera_fps,(time_window[1]+abs(time_window[0]))*camera_fps])
        axes2.set_xticklabels([time_window[0],0,time_window[1]])
        axes2.spines['left'].set_linewidth(2) 
        fix_ylim=True
        # if event_of_interest.startswith('stop'):
        #     yrange=[0,5]
        # else:
        yrange=[0,10]
        if fix_ylim:
            axes.set_ylim(yrange)
            axes.set_yticks(yrange)
            axes2.set_ylim([0, 1])
            axes2.set_yticks([0,1])
        png_name = f"{event_of_interest}_ts_plot.png"
        fig.savefig(oe_folder / png_name)
        svg_name = f"{event_of_interest}_ts_plot.svg"
        fig.savefig(oe_folder / svg_name)
    else:
        print('analyse spiking activity without behaviour')


    ## needs to reorganise this part. Figure out a better to plot data about behavioural-state modulated stimulus response
    # if velocity_file is not None and trial_file is not None and analyse_behavioural_state_modulation==True:
    #     #velocity_tbt = np.load(velocity_file)
    #     behavioural_trial_type=['walking_trials','stationary_trials','straight_walk_trials','turning_trials']
    #     walking_trials,stationary_trials,straight_walk_trials,turning_trials=classify_trial_type(meta_info, 20, 100)
    #     if event_of_interest.lower() in behavioural_trial_type:
    #         if event_of_interest.lower() == 'walking_trials':
    #             metrics_to_classify=walking_trials.index
    #         elif event_of_interest.lower() == 'stationary_trials':
    #             metrics_to_classify=stationary_trials.index
    #         elif event_of_interest.lower() == 'straight_walk_trials':
    #             metrics_to_classify=straight_walk_trials.index
    #         elif event_of_interest.lower() == 'turning_trials':
    #             metrics_to_classify=turning_trials.index
    #         if metrics_to_classify.shape[0]>0:
    #             event_times_of_interest=events_time[metrics_to_classify]
    #         else:
    #             print("event of interest does not present in this animal. Analyse the data without behavioural inputs")
    #             event_times_of_interest = events_time
    #     else:
    #         print("only 'walking_trials','stationary_trials','straight_walk_trials','turning_trials' can be used for behavioural trial type . Analyse the data without behavioural inputs")
    #         event_times_of_interest = events_time
    # else:
    #     event_times_of_interest = events_time


    ###start loading info from sorted spikes
    ## if use kilosort standalone, then load kilosort folder. Otherwise, load spikeinterface's preprocessed data and its toolkit.
    if analysis_methods.get("motion_corrector")=="kilosort_default" or analysis_methods.get("motion_corrector")=="testing":
        #main_foler_name='kilosort4_ThU13_ThL11'
        #main_foler_name='kilosort4'
        main_foler_name='kilosort4_motion_corrected'
        #main_foler_name='kilosort4_ThU18_ThL17_T0_T1500'
        #main_foler_name='kilosort4_T0_T1500'
        merged_units=True
        folder_suffix="_merged" if merged_units else ""
        file_type=".npy"
        ks_path=oe_folder/f"{main_foler_name}{folder_suffix}"/ "shank_0"
        if ks_path.is_dir():
            pass
        else:
            ks_path= oe_folder/f"{main_foler_name}{folder_suffix}"
        spike_clusters=np.load(ks_path/"spike_clusters.npy")
        spike_times=np.load(ks_path/"spike_times.npy")/30000.0#this is the default sampling frequency in openEphys
        cluster_group=pd.read_csv(ks_path/"cluster_group.tsv", sep='\t',header=0)
        if analysis_methods.get("include_MUA") == True:
            mask= np.isin(spike_clusters,cluster_group.loc[(cluster_group['group'].reset_index(drop=True)=='mua') | (cluster_group['group'].reset_index(drop=True)=='good')].values)
        else:
            mask= np.isin(spike_clusters,cluster_group.loc[cluster_group['group']=='good']['cluster_id'].values)

        cluster_id_interest=spike_clusters[mask]
        spike_time_interest=spike_times[mask]
        for this_unit in np.unique(cluster_id_interest):
            spike_time_temp=spike_time_interest[cluster_id_interest==this_unit]
            #duplicated_spikes=sc.find_duplicated_spikes(spike_time_temp,censored_period=0.0001,seed=0)
            duplicated_spikes=sc.find_duplicated_spikes(spike_time_temp,censored_period=0.0015,seed=0)#we had to introduce a refractory period (tref = 1.5 ms) to replicate the nonlinear relation between the peak instantaneous firing rate frequency f0 and current over a large fraction of the LGMD firing range (Fig. 5B, gray trace).
            spike_time_interest=np.delete(spike_time_interest, duplicated_spikes)
            cluster_id_interest=np.delete(cluster_id_interest, duplicated_spikes)
    else:
        recording_saved = get_preprocessed_recording(oe_folder,analysis_methods)
        if (
            analysis_methods.get("load_analyser_from_disc") == True
            and (oe_folder / analyser_folder_name).is_dir()
        ):
            sorting_analyzer = si.load_sorting_analyzer(
                folder=oe_folder / analyser_folder_name
            )
            print(f"{sorting_analyzer}")
            recording_saved = sorting_analyzer.recording
            sorting_spikes = sorting_analyzer.sorting
            unit_labels = sorting_spikes.get_property("quality")
        else:
            if analysis_methods.get("include_MUA") == True:
                cluster_group_interest = ["noise"]
            else:
                cluster_group_interest = ["noise", "mua"]
            sorting_spikes = se.read_phy(
                oe_folder / phy_folder_name, exclude_cluster_groups=cluster_group_interest
            )
            unit_labels = sorting_spikes.get_property("quality")
            sorting_analyzer = si.create_sorting_analyzer(
                sorting=sorting_spikes,
                recording=recording_saved,
                sparse=False, # default
                format="memory",  # default
            )
            calculate_analyzer_extension(sorting_analyzer)
            # sqm.compute_quality_metrics(sorting_analyzer,metric_names=["isolation_distance","d_prime"])
            # sw.plot_quality_metrics(sorting_analyzer, include_metrics=["amplitude_cutoff", "presence_ratio", "isi_violations_ratio", "snr","isolation_distance","d_prime"])

    ## go through the peri_event_time_histogram of every cluster
        spike_time_interest, cluster_id_interest, _, _ = spike_overview(
            oe_folder,
            this_sorter,
            sorting_spikes,
            sorting_analyzer,
            recording_saved,
            unit_labels,
        )
        print(f"printing an overview of spikes detected by {this_sorter} from {oe_folder}")

    # sort spike time for the function get_spike_counts_in_bins
    spike_time_interest_sorted, cluster_id_interest_sorted = sort_arrays(
        spike_time_interest, cluster_id_interest
    )
    spike_count, cluster_id = get_spike_counts_in_bins(
        spike_time_interest_sorted, cluster_id_interest_sorted, events_time_tw
    )

    ## go through the peri_event_time_histogram of every cluster
    if "stim_type" in locals():
        print("use stim type generated by sorting trial info")
    else:
        stim_type = analysis_methods.get("stim_type")
    ## it might be useful to calulcate all the clusters together but right now it does not look necessary
    # peth_means, peth_stds, tscale,ids=calculate_peths_details(
    # spike_time_interest,cluster_id_interest, np.unique(cluster_id_interest), events_time, pre_time=abs(time_window[0]),
    # post_time=time_window[1], bin_size=0.025, smoothing=0.025, return_fr=True)
    # for i, this_id in enumerate(ids):
    #     fig2, (ax1,ax2) = plt.subplots(nrows=2, figsize=(12, 8), sharex=True)
    #     mean=peth_means[i,:]
    #     bars=peth_stds[i,:]
    #     ax1.plot(tscale,mean, linewidth=3, color="blue")
    #     negative_std=mean - bars
    #     if np.any(negative_std<0):            
    #         ax1.fill_between(tscale,0, mean + bars, color= 'blue', alpha= 0.5)
    #     else:
    #         ax1.fill_between(tscale,negative_std, mean + bars,color= 'blue', alpha= 0.5)
    #     ax1.set(ylabel="Rate (spikes/sec)",xlim=(time_window[0], time_window[1]))
    #     ax1.axvline(0.0,color="black")
    #     #ax2.plot(peth.to_tsd(), "|", markersize=1, color="black", mew=1)
    #     #ax2.set(ylabel="Event #",xlabel=f"Time from {event_of_interest} (s)",xlim=(time_window[0], time_window[1]))
    #     fix_ylim=False
    #     if fix_ylim:
    #         ax1.set_ylim([0, 250])
    #         ax1.set_yticks([0,250])
    #     cleanup_xticks=False
    #     if cleanup_xticks:
    #         ax2.set_xticks([time_window[0],round(time_window[0]/2),0,round(time_window[1]/2),time_window[1]])
    #     png_name = f"unit{this_id}_{event_of_interest}_peth_test.png"
    #     fig2.savefig(oe_folder / png_name)
    ####
    ## similiar idea but use pynapple package
    ####
    # spike_dict={}
    # ids=np.unique(cluster_id_interest)
    # for keys in ids:
    #     spike_dict[keys] = spike_time_interest[np.where(cluster_id_interest==keys)[0]]
    # peth = nap.compute_perievent(
    # timestamps=nap.TsGroup(spike_dict,time_units="s"),
    # tref=nap.Ts(t=events_time, time_units="s"), 
    # minmax=(time_window[0], time_window[1]), 
    # time_unit="s")
    # for this_id in ids:
    #     fig2, (ax1,ax2) = plt.subplots(nrows=2, figsize=(12, 8), sharex=True)
    #     mean=np.mean(peth[this_id].count(0.05), 1) / 0.05
    #     #bars=np.std(peth[this_id].count(0.05), 1) / 0.05
    #     ax1.plot(mean, linewidth=3, color="red")
    #     #ax1.plot(tscale,mean, linewidth=3, color="blue")
    #     # negative_std=mean - bars
    #     # if np.any(negative_std<0):            
    #     #     ax1.fill_between(tscale,0, mean + bars, color= 'blue', alpha= 0.5)
    #     # else:
    #     #     ax1.fill_between(tscale,negative_std, mean + bars,color= 'blue', alpha= 0.5)
    #     ax1.set(ylabel="Rate (spikes/sec)",xlim=(time_window[0], time_window[1]))
    #     ax1.axvline(0.0,color="black")
    #     ax2.plot(peth[this_id].to_tsd(), "|", markersize=1, color="black", mew=1)
    #     ax2.set(ylabel="Event #",xlabel=f"Time from {event_of_interest} (s)",xlim=(time_window[0], time_window[1]))
    #     fix_ylim=False
    #     if fix_ylim:
    #         ax1.set_ylim([0, 250])
    #         ax1.set_yticks([0,250])
    #     cleanup_xticks=False
    #     if cleanup_xticks:
    #         ax2.set_xticks([time_window[0],round(time_window[0]/2),0,round(time_window[1]/2),time_window[1]])
    #     png_name = f"unit{this_id}_{event_of_interest}_peth_naptest.png"
    #     fig2.savefig(oe_folder / png_name)


    for this_cluster_id in np.unique(cluster_id_interest):
        if analysis_methods.get("analysis_by_stimulus_type") == True and trial_file is not None:
            for this_variable in meta_info[stim_variable2].unique():
                for this_stim in stim_type:
                    if np.where((meta_info["stim_type"] == this_stim) & (meta_info[stim_variable2]==this_variable))[0].shape[0]<2:
                        continue
                    ax = peri_event_time_histogram(
                        spike_time_interest,
                        cluster_id_interest,
                        events_time[
                            (meta_info["stim_type"] == this_stim) & (meta_info[stim_variable2]==this_variable)
                        ],
                        this_cluster_id,
                        t_before=abs(time_window[0]),
                        t_after=time_window[1],
                        include_raster=True,
                        raster_kwargs={"color": "black", "lw": 0.5},
                    )
                    fix_ylim=False
                    if fix_ylim:
                        ax.set_ylim([0, 250])
                        ax.set_yticks([0,250])
                        ax.set_xticks([])
                        ax.set_xlabel("")
                        ax.set_ylabel("")
                        jpg_name = f"unit{this_cluster_id}_peth_stim{this_stim}_{stim_variable2}_{this_variable}_no_raster.png"
                        #svg_name = f"unit{this_cluster_id}_peth_stim{this_stim}_{stim_variable2}_{this_variable}_no_raster.svg"
                        ax.figure.savefig(oe_folder / jpg_name)
                    else:
                        #jpg_name = f"unit{this_cluster_id}_peth_stim{this_stim}_{stim_variable2}_{this_variable}.png"
                        svg_name = f"unit{this_cluster_id}_peth_stim{this_stim}_{stim_variable2}_{this_variable}.svg"
                        ax.figure.savefig(oe_folder / svg_name)     
        else:
            these_spikes=spike_time_interest[np.where(cluster_id_interest==this_cluster_id)[0]]
            these_ids=np.ones(these_spikes.shape[0],dtype=int)*this_cluster_id
            tspikes=nap.Tsd(
            t=these_spikes, 
            d=these_ids, time_units="s")           
            
            peth = nap.compute_perievent(
            timestamps=tspikes,
            tref=nap.Ts(t=events_time, time_units="s"), 
            minmax=(time_window[0], time_window[1]), 
            time_unit="s")
            
            peth_means, peth_stds, tscale,_=calculate_peths_details(
                these_spikes,these_ids, [this_cluster_id], events_time, pre_time=abs(time_window[0]),
                post_time=time_window[1], bin_size=0.025, smoothing=0.025, return_fr=True)
            fig2, (ax1,ax2) = plt.subplots(nrows=2, figsize=(9, 6), sharex=True)
            #ax1.plot(np.mean(peth.count(0.05), 1) / 0.05, linewidth=3, color="red")
            mean=peth_means[0,:]
            bars=peth_stds[0,:]
            ax1.plot(tscale,mean, linewidth=3, color="blue")
            negative_std=mean - bars
            if np.any(negative_std<0):            
                ax1.fill_between(tscale,0, mean + bars, color= 'blue', alpha= 0.5)
            else:
                ax1.fill_between(tscale,negative_std, mean + bars,color= 'blue', alpha= 0.5)
            ax1.set(ylabel="Rate (spikes/sec)",xlim=(time_window[0], time_window[1]))
            ax1.axvline(0.0,color="black")
            ax2.plot(peth.to_tsd(), "|", markersize=1, color="black", mew=1)
            ax2.set(ylabel="Event #",xlabel=f"Time from {event_of_interest} (s)",xlim=(time_window[0], time_window[1]))
            fix_ylim=True
            if fix_ylim:
                ax1.set_ylim([0, 90])
                ax1.set_yticks([0,90])
            cleanup_xticks=True
            if cleanup_xticks:
                ax2.set_xticks([time_window[0],round(time_window[0]/2),0,round(time_window[1]/2),time_window[1]])
            png_name = f"unit{this_cluster_id}_{event_of_interest}_peth.png"
            fig2.savefig(oe_folder / png_name)
    json_string = json.dumps(analysis_methods, indent=1)
    with open(oe_folder / "ephys_analysis_methods_backup.json", "w") as f:
        f.write(json_string)
    return spike_count, cluster_id


if __name__ == "__main__":
    #thisDir = r"Y:\GN25009\250403\coherence\session1\2025-04-03_19-13-57"
    #thisDir = r"Y:\GN25017\250518\gratings\session1\2025-05-18_21-32-15"
    #thisDir = r"Y:\GN25028\250727\coherence\session1\2025-07-27_19-24-54"
    #thisDir = r"Y:\GN25029\250729\sweeping\session1\2025-07-29_16-34-15"
    #thisDir = r"Y:\GN25029\250729\coherence\session1\2025-07-29_20-16-03"
    #thisDir = r"Y:\GN25029\250729\looming\session3\2025-07-29_18-35-50"
    #thisDir = r"Y:\GN25029\250729\looming\session1\2025-07-29_15-22-54"
    #thisDir = r"Y:\GN25034\250907\looming\session2\2025-09-07_21-18-07"
    #thisDir = r"Y:\GN25049\251025\looming\session2\2025-10-25_18-23-55"
    #thisDir = r"Y:\GN25049\251025\looming\session1\2025-10-25_16-06-08"
    #thisDir = r"Y:\GN25049\251025\looming\session4\2025-10-25_21-53-25"
    #thisDir = r"Y:\GN25049\251025\looming\session3\2025-10-25_20-08-11"
    #thisDir = r"Y:\GN25034\250907\sweeping\session2\2025-09-07_22-44-33"
    #thisDir = r"Y:\GN25039\250927\looming\session1\2025-09-27_14-44-46"
    #thisDir = r"Y:\GN25040\250928\looming\session1\2025-09-28_15-55-12"
    #thisDir = r"Y:\GN25041\251004\looming\session1\2025-10-04_16-39-59"
    #thisDir = r"Y:\GN25042\251005\looming\session1\2025-10-05_16-22-44"
    #thisDir = r"Y:\GN25051\251101\gratings\session1\2025-11-01_20-31-41"
    #thisDir = r"Y:\GN25051\251101\looming\session1\2025-11-01_16-35-52"
    #thisDir = r"Y:\GN25051\251101\sweeping\session1\2025-11-01_18-24-40"
    #thisDir = r"Y:\GN25048\251019\looming\session1\2025-10-19_18-50-34"
    #thisDir = r"Y:\GN25044\251012\looming\session1\2025-10-12_14-22-01"
    #thisDir = r"Y:\GN25030\250802\looming\session1\2025-08-02_19-34-32"
    #thisDir = r"Y:\GN25033\250906\looming\session1\2025-09-06_18-42-24"
    #thisDir = r"Y:\GN25037\250922\looming\session1\2025-09-22_13-24-39"
    #thisDir = r"Y:\GN25038\250924\looming\session1\2025-09-24_21-10-39"
    #thisDir = r"Y:\GN25049\251026\looming\session1\2025-10-25_16-06-08"
    #thisDir = r"Y:\GN25049\251026\looming\session3\2025-10-25_20-08-11"
    #thisDir = r"Y:\GN25035\250913\looming\session1\2025-09-13_14-29-00"
    #thisDir = r"Y:\GN25034\250907\looming\session4\2025-09-08_04-05-44"
    #thisDir = r"Y:\GN25045\251013\looming\session1\2025-10-13_11-16-41"
    #thisDir = r"Y:\GN25046\251018\looming\session1\2025-10-18_16-34-27"
    #thisDir = r"Y:\GN25045\251013\looming\session2\2025-10-13_13-31-57"
    #thisDir = r"Y:\GN25043\251008\looming\session2\2025-10-07_14-22-22"
    #thisDir = r"Y:\GN25037\250922\looming\session2\2025-09-22_15-59-41"
    #thisDir = r"Y:\GN25031\250803\looming\session1\2025-08-03_17-52-45"
    #thisDir = r"Y:\GN25037\250922\looming\session3\2025-09-22_17-48-42"
    #thisDir = r"Y:\GN25039\250927\looming\session2\2025-09-27_17-12-17"
    #thisDir = r"Y:\GN25032\250807\looming\session1\2025-08-07_19-34-42"
    #thisDir = r"Y:\GN25032\250807\looming\session2\2025-08-07_22-06-12"
    #thisDir = r"Y:\GN25032\250807\looming\session4\2025-08-08_01-19-05"
    #thisDir = r"Y:\GN25029\250729\looming\session2\2025-07-29_17-35-20"
    #thisDir = r"Y:\GN25060\251130\coherence\session1\2025-11-30_14-25-01"
    #thisDir = r"Y:\GN25049\251025\looming\session5\2025-10-25_23-33-49"
    #thisDir = r"Y:\GN25053\251108\looming\session2\2025-11-08_15-39-42"
    #thisDir = r"Y:\GN25051\251101\looming\session2\2025-11-01_22-39-06"
    #thisDir = r"Y:\GN25045\251013\looming\session2\2025-10-13_13-31-57"
    #thisDir = r"Y:\GN25065\251214\flashing\session1\2025-12-14_13-48-09"
    #thisDir = r"Y:\GN25063\251213\flashing\session1\2025-12-13_16-03-57"
    #thisDir = r"Y:\GN25068\251221\flashing\session1\2025-12-21_14-44-50"
    #thisDir = r"Y:\GN25067\251220\flashing\session1\2025-12-20_14-36-54"
    #thisDir = r"Y:\GN25066\251214\sweeping\session1\2025-12-14_20-35-57"
    #thisDir = r"Y:\GN26008\250727\spontaneous\session1\2026-01-25_14-33-17"
    thisDir = r"Y:\GN26012\260208\spontaneous\session1\2026-02-08_13-56-52"
    #thisDir = r"Y:\GN26011\260207\spontaneous\session1\2026-02-07_13-28-12"
    #thisDir = r"Y:\GN25065\251214\sweeping\session1\2025-12-14_14-14-29"
    #thisDir = r"Y:\GN25060\251130\looming\session1\2025-11-30_16-12-19"
    json_file = "./analysis_methods_dictionary.json"

    ##Time the function
    tic = time.perf_counter()
    align_async_signals(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
