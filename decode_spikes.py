import time, os, json, warnings, sys
from open_ephys.analysis import Session
import spikeinterface.core as si
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
from spikeinterface.widgets import plot_sorting_summary
import math
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.plot import peri_event_time_histogram, driftmap_color, driftmap
from brainbox.ephys_plots import (
    plot_cdf,
    image_rms_plot,
    scatter_raster_plot,
    scatter_amp_depth_fr_plot,
    probe_rms_plot,
)

import numpy as np
from pathlib import Path
import pandas as pd
from extraction_barcodes_cl import extract_barcodes
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
from data_cleaning import sorting_trial_info

# sys.path.insert(0, str(parent_dir) + "\\bonfic")
# from analyse_stimulus_evoked_response import classify_trial_type
n_cpus = os.cpu_count()
n_jobs = n_cpus - 4
global_job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)
# global_job_kwargs = dict(n_jobs=16, chunk_duration="5s", progress_bar=False)
si.set_global_job_kwargs(**global_job_kwargs)


def root_sum_squared(tuples):
    result = []
    for tup in tuples:
        # Sum of squares of tuple elements
        sum_of_squares = sum(x**2 for x in tup)
        # Add square root of sum of squares to result
        result.append(math.sqrt(sum_of_squares))
    return result


def classify_walk(arr, speed_threshold=10, on_consecutive_length=50):
    walk_events_all = []
    until_walk_events = []
    from_walk_events = []
    mask = arr > speed_threshold
    trans = np.diff(mask)
    where_transition = np.where(trans)
    for i in np.unique(where_transition[0]):
        trial_of_interest = where_transition[1][where_transition[0] == i]
        w2s = trial_of_interest[arr[i, trial_of_interest] > speed_threshold]
        s2w = trial_of_interest[arr[i, trial_of_interest] < speed_threshold]
        if len(w2s) == len(s2w) and arr[i, 0] < speed_threshold:
            until_walk_frame = w2s[w2s - s2w > on_consecutive_length]
            from_walk_frame = s2w[abs(s2w - w2s) > on_consecutive_length]
        elif len(w2s) == len(s2w) and arr[i, 0] > speed_threshold:
            s2w = np.insert(0, 1, s2w)
            w2s = np.insert(arr.shape[1] - 1, -1, w2s)
            until_walk_frame = w2s[w2s - s2w > on_consecutive_length]
            from_walk_frame = s2w[abs(s2w - w2s) > on_consecutive_length]
        elif len(s2w) > len(w2s) and arr[i, 0] < speed_threshold:
            w2s = np.insert(arr.shape[1] - 1, -1, w2s)
            until_walk_frame = w2s[w2s - s2w > on_consecutive_length]
            from_walk_frame = s2w[abs(s2w - w2s) > on_consecutive_length]
        elif len(s2w) < len(w2s) and arr[i, 0] > speed_threshold:
            s2w = np.insert(0, 1, s2w)
            until_walk_frame = w2s[w2s - s2w > on_consecutive_length]
            from_walk_frame = s2w[abs(s2w - w2s) > on_consecutive_length]

        # if len(from_walk_frame) == 0 and len(from_walk_frame) == 0:
        #     continue
        # elif len(from_walk_frame) == 0:
        #     walk_events = [i, 0, until_walk_frame]
        # elif len(until_walk_frame) == 0:
        #     walk_events = [i, from_walk_frame, arr.shape[1] - 1]
        # else:
        #     walk_events = [i, from_walk_frame, until_walk_frame]
        # walk_events_all.append(walk_events)
        if len(from_walk_frame) == 0 and len(from_walk_frame) == 0:
            continue
        elif len(from_walk_frame) == 0:
            until_walk_events.append(np.array([i, until_walk_frame]))
        elif len(until_walk_frame) == 0:
            from_walk_events.append(np.array([i, from_walk_frame]))
        else:
            until_walk_events.append(
                np.vstack(
                    [np.ones(len(until_walk_frame), dtype=int) * i, until_walk_frame]
                )
            )
            from_walk_events.append(
                np.vstack(
                    [np.ones(len(from_walk_frame), dtype=int) * i, from_walk_frame]
                )
            )
        # walk_events_all.append(walk_events)

    return np.hstack(from_walk_events), np.hstack(until_walk_events)
    # return walk_events_all


# def classify_walk_copilot(arr, speed_threshold=10, on_consecutive_length=30):
#     walk_events_all = []
#     mask = arr > speed_threshold
#     trans = np.diff(mask.astype(int), axis=1)
#     where_transition = np.where(trans)

#     for i in np.unique(where_transition[0]):
#         trial_of_interest = where_transition[1][where_transition[0] == i]
#         w2s = trial_of_interest[arr[i, trial_of_interest] > speed_threshold]
#         s2w = trial_of_interest[arr[i, trial_of_interest] <= speed_threshold]

#         if arr[i, 0] > speed_threshold:
#             w2s = np.append(w2s, arr.shape[1] - 1)
#         else:
#             s2w = np.append(s2w, arr.shape[1] - 1)

#         until_walk_frame = []
#         from_walk_frame = []
#         for j in range(min(len(w2s), len(s2w))):
#             if w2s[j] - s2w[j] >= on_consecutive_length:
#                 until_walk_frame.append(w2s[j])
#                 from_walk_frame.append(s2w[j])

#         walk_events = [
#             [i, from_walk_frame[k], until_walk_frame[k]]
#             for k in range(len(from_walk_frame))
#         ]
#         walk_events_all.extend(walk_events)

#     return walk_events_all
# kernel = np.ones(consecutive_length, dtype=int)
# if arr.ndim == 1:
#     # Create a kernel of length 5 filled with ones

#     # Use convolution to find the sum of 5 consecutive elements
#     conv = np.convolve(mask, kernel, mode="valid")

#     # Find the indices where the convolution result is 5 or more
#     start_indices = np.where(conv >= consecutive_length)[0]

#     # Adjust the indices to ensure that they point to the start of 5 or more consecutive elements
#     start_indices = [
#         idx for idx in start_indices if np.all(mask[idx : idx + consecutive_length])
#     ]
#     events = np.asarray(start_indices)
# elif arr.ndim > 1:
#     conv = np.apply_along_axis(
#         lambda m: np.convolve(m, kernel, "valid"),
#         axis=1,
#         arr=mask,
#     )
#     # Find where the convolution is greater than or equal to 5 (indicating at least 5 consecutive True values)
#     events = np.argwhere(conv >= consecutive_length)
#     events = np.array(
#         [
#             (idx[0], idx[1])
#             for idx in events
#             if idx[1] + consecutive_length < arr.shape[1]
#         ]
#     )
# else:
#     print("this array is empty")
#     events = None
# return events


def remove_run_during_isi(camera_time, camera_fps, ISI_duration):
    isi_len = camera_fps * ISI_duration
    during_stim_event = np.vstack(
        [
            camera_time[0, :][camera_time[1, :] - isi_len > 0],
            camera_time[1, :][camera_time[1, :] - isi_len > 0],
        ]
    )
    # during_stim_event = camera_time[np.where([camera_time[1, :] - isi_len > 0])[1]]
    return during_stim_event


def estimating_ephys_timepoints(
    camera_time, ephys_time, camera_fps, stim_duration, ISI_duration
):
    stim_len = camera_fps * stim_duration
    isi_len = camera_fps * ISI_duration
    stim_len_oe = np.diff(ephys_time)
    time_from_stim = np.multiply(
        (camera_time[1, :] - isi_len) / stim_len,
        stim_len_oe[camera_time[0, :]],
    )
    events = ephys_time[camera_time[0, :]] + time_from_stim
    return events


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


def align_async_signals(thisDir, json_file):
    colormap_name = "coolwarm"
    COL = MplColorHelper(colormap_name, 0, 8)
    oe_folder = Path(thisDir)
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    this_sorter = analysis_methods.get("sorter_name")
    this_experimenter = analysis_methods.get("experimenter")
    sorter_suffix = generate_sorter_suffix(this_sorter)
    result_folder_name = "results" + sorter_suffix
    sorting_folder_name = "sorting" + sorter_suffix
    analyser_folder_name = "analyser" + sorter_suffix
    phy_folder_name = phy_folder_name = "phy" + sorter_suffix
    report_folder_name = "report" + sorter_suffix

    ##load adc events from openEphys
    session = Session(oe_folder)
    recording = session.recordnodes[0].recordings[0]
    camera_trigger_on_oe = recording.events.timestamp[
        (recording.events.line == 2) & (recording.events.state == 1)
    ]
    # barcode_on_oe = recording.events.timestamp[
    #     (recording.events.line == 3) & (recording.events.state == 1)
    # ]

    barcode_on_oe = recording.events.sample_number[recording.events.line == 3]
    if len(barcode_on_oe) > 0:
        _, signals_time_and_bars_array = extract_barcodes(
            oe_folder, barcode_on_oe.values
        )
    stim_directory = oe_folder.resolve().parents[0]
    database_ext = "database*.pickle"
    tracking_file = find_file(stim_directory, database_ext)
    stim_sync_ext = "*_stim_sync.csv"
    stim_sync_file = find_file(stim_directory, stim_sync_ext)
    barcode_ext = "*_barcode.csv"
    barcode_file = find_file(stim_directory, barcode_ext)
    velocity_ext = "*velocity.npy"
    velocity_file = find_file(stim_directory, velocity_ext)
    rotation_ext = "z_vector.npy"
    rotation_file = find_file(stim_directory, rotation_ext)
    camera_fps = analysis_methods.get("camera_fps")
    walk_speed_threshold = 20
    walk_consecutive_length = int(camera_fps * 0.5)
    event_of_interest = analysis_methods.get("event_of_interest")

    if analysis_methods.get("analyse_stim_evoked_activity") == True:
        stim_duration = analysis_methods.get("stim_duration")
        ISI_duration = analysis_methods.get("interval_duration")
        walking_trials_threshold = 50
        turning_trials_threshold = 0.33
        time_window = np.array([-0.1, 0.0])
        time_window_behaviours = np.array([-1, 2])
        ##load stimulus meta info
        pd_ext = "behavioural_summary.pickle"
        stimulus_meta_file = find_file(stim_directory, pd_ext)

        if stimulus_meta_file is None:
            if oe_folder.name == "2024-02-01_15-25-25":
                stim_directory = Path(
                    r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23018\240422\coherence\session2"
                )
            print("load raw stimulus information")
            trial_ext = "trial*.csv"
            this_csv = find_file(stim_directory, trial_ext)
            stim_directory = pd.read_csv(this_csv)
            meta_info, _ = sorting_trial_info(stim_directory)
            stimulus_meta_info = meta_info[1::2]
            num_stim = stimulus_meta_info.shape[0]
        else:
            stimulus_meta_info = pd.read_pickle(stimulus_meta_file)
            num_stim = stimulus_meta_info.shape[0]

        isi_on_oe = recording.events.timestamp[
            (recording.events.line == 1) & (recording.events.state == 1)
        ]
        stim_on_oe = recording.events.timestamp[
            (recording.events.line == 1) & (recording.events.state == 0)
        ]

        if analysis_methods.get("analyse_stim_evoked_activity") == True:
            if len(stim_on_oe) > num_stim:
                stim_events_times = stim_on_oe[
                    -num_stim:
                ].values  ##this happens when the S button is pressed after openEphys are recorded.
            else:
                stim_events_times = stim_on_oe[:].values
        elif event_of_interest.lower() == "preStim_ISI":
            stim_events_times = isi_on_oe[1:].values
        elif event_of_interest.lower() == "postStim_ISI":
            stim_events_times = isi_on_oe[:-1].values
        else:
            (
                "Not found what properties of stimuli you want to analyse. Double check event of interest"
            )
            return None
        ##build up analysis time window
        stim_events_tw = np.array(
            [stim_events_times + time_window[0], stim_events_times + time_window[1]]
        ).T
        if event_of_interest.lower().startswith("walk"):
            if velocity_file is None:
                print(
                    "no velocity across frames avaliable. Use behavioural summary for responses from each trial"
                )
                csv_ext = "trial*.csv"
                this_csv = find_file(stim_directory, csv_ext)
                stim_directory = pd.read_csv(this_csv)
                num_stim = int(stim_directory.shape[0] / 2)
                walking_trials_threshold = 50
                turning_trials_threshold = 0.33
                classify_walk(
                    stimulus_meta_info,
                    turning_trials_threshold,
                    walking_trials_threshold,
                )
            else:
                velocity_tbt = np.load(velocity_file)
                walk_events_start, walk_events_end = classify_walk(
                    velocity_tbt[:, 0 : (stim_duration + ISI_duration) * camera_fps],
                    walk_speed_threshold,
                    walk_consecutive_length,
                )
                # walk_events_start = remove_run_during_isi(
                #     walk_events_start, camera_fps, ISI_duration
                # )
                walk_events_start_tw = np.array(
                    [
                        walk_events_start[1, :]
                        + time_window_behaviours[0] * camera_fps,
                        walk_events_start[1, :]
                        + time_window_behaviours[1] * camera_fps,
                    ]
                ).T
                walk_events_end_tw = np.array(
                    [
                        walk_events_end[1, :] + time_window_behaviours[0] * camera_fps,
                        walk_events_end[1, :] + time_window_behaviours[1] * camera_fps,
                    ]
                ).T
                fig1, (ax, ax1) = plt.subplots(
                    nrows=2, ncols=1, figsize=(18, 7), tight_layout=True
                )
                for i in range(0, walk_events_start.shape[1]):
                    velocity_of_interest = velocity_tbt[
                        walk_events_start[0, i],
                        walk_events_start_tw[i, 0] : walk_events_start_tw[i, 1],
                    ]
                    ax.plot(
                        range(0, len(velocity_of_interest)),
                        velocity_of_interest,
                        linewidth=7.0,
                    )

                ax.set_xlabel("Time")
                ax.set_ylabel("Velocity")
                ax.set_ylim(0, 300)
                plot_name = "walk_onset.svg"
                if analysis_methods.get("debug_mode") == False:
                    fig1.savefig(Path(stim_directory) / plot_name)
                plt.show()
                if len(barcode_on_oe) == 0 and len(camera_trigger_on_oe) == 0:
                    walk_events_start_oe = estimating_ephys_timepoints(
                        walk_events_start,
                        stim_events_times,
                        camera_fps,
                        stim_duration,
                        ISI_duration,
                    )
                else:
                    print(
                        "work in progress. Here I need to output walk event starts based on oe time"
                    )
                walk_events_start_oe_tw = np.array(
                    [
                        walk_events_start_oe + time_window[0],
                        walk_events_start_oe + time_window[1],
                    ]
                ).T
    else:
        print(
            "this part needs more work. basically we should just detect whether there is info about stimulation. If not just skip aligning behaviours with certain stimulus"
        )

    if event_of_interest.lower().startswith("stim"):
        event_of_interest = stim_events_times
        event_of_interest_tw = stim_events_tw
    elif event_of_interest.lower() == "walk_onset":
        event_of_interest = walk_events_start_oe
        event_of_interest_tw = walk_events_start_oe_tw
        print("Align spikes with the onset of walk events")
    elif event_of_interest.lower() == "walk_offset":
        print("Align spikes with the offset of walk events")
        print("Work in progress")
        return

    ###start loading info from sorted spikes
    recording_saved = get_preprocessed_recording(oe_folder)
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
        recording_saved = get_preprocessed_recording(oe_folder)
        sorting_analyzer = si.create_sorting_analyzer(
            sorting=sorting_spikes,
            recording=recording_saved,
            sparse=True,  # default
            format="memory",  # default
        )
        calculate_analyzer_extension(sorting_analyzer)

    ## go through the peri_event_time_histogram of every cluster
    spike_time_all, cluster_id_all, spike_amp_all, spike_loc_all = spike_overview(
        oe_folder,
        this_sorter,
        sorting_spikes,
        sorting_analyzer,
        recording_saved,
        unit_labels,
    )
    print(f"printing an overview of spikes detected by {this_sorter} from {oe_folder}")

    # sort spike time for the function get_spike_counts_in_bins
    spike_time_all_sorted, cluster_id_all_sorted = sort_arrays(
        spike_time_all, cluster_id_all
    )
    spike_count, cluster_id = get_spike_counts_in_bins(
        spike_time_all_sorted, cluster_id_all_sorted, stim_events_tw
    )
    # work in progress: testing how to use this function
    # scatter_amp_depth_fr_plot(
    #     spike_amps=spike_amp_all,
    #     spike_clusters=cluster_id_all,
    #     spike_depths=spike_loc_all,
    #     spike_times=spike_time_all,display=True
    # )

    ## go through the peri_event_time_histogram of every cluster
    for this_cluster_id in np.unique(cluster_id_all):
        if analysis_methods.get("analysis_by_stimulus_type") == True:
            stim_type = analysis_methods.get("stim_type")
            for this_stim in stim_type:

                # troubleshoot this part. Dont know why it exceeds the max size
                ax = peri_event_time_histogram(
                    spike_time_all,
                    cluster_id_all,
                    event_of_interest[
                        stimulus_meta_info.loc[:, "stim_type"] == this_stim
                    ],
                    this_cluster_id,
                    t_before=abs(time_window_behaviours[0]),
                    t_after=time_window_behaviours[1],
                    bin_size=0.025,
                    smoothing=0.025,
                    include_raster=True,
                    raster_kwargs={"color": "black", "lw": 1},
                )
                fig_name = f"peth_stim{this_stim}_unit{this_cluster_id}.svg"
                fig_dir = oe_folder / fig_name
                ax.figure.savefig(fig_dir)
        else:
            ax = peri_event_time_histogram(
                spike_time_all,
                cluster_id_all,
                event_of_interest,
                this_cluster_id,
                t_before=abs(time_window_behaviours[0]),
                t_after=time_window_behaviours[1],
                bin_size=0.025,
                smoothing=0.025,
                include_raster=False,
                raster_kwargs={"color": "black", "lw": 1},
            )
    return spike_count, cluster_id


if __name__ == "__main__":
    # thisDir = r"C:\Users\neuroLaptop\Documents\Open Ephys\P-series-32channels\GN00003\2023-12-28_14-39-40"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23019\240507\coherence\session1\2024-05-07_23-08-55"
    thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23018\240422\coherence\session2\2024-04-22_01-09-50"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23015\240201\coherence\session1\2024-02-01_15-25-25"
    # thisDir = r"C:\Users\neuroPC\Documents\Open Ephys\2024-02-01_15-25-25"
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    align_async_signals(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
