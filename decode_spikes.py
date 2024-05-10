import time, os, json, warnings, sys
import spikeinterface.full as sf
from open_ephys.analysis import Session
import spikeinterface.core as si
import spikeinterface.extractors as se
import matplotlib.pyplot as plt
import spikeinterface.qualitymetrics as sqm
from spikeinterface.widgets import plot_sorting_summary
import numcodecs
"""
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.plot import peri_event_time_histogram
from brainbox.ephys_plots import (
    plot_cdf,
    image_rms_plot,
    scatter_raster_plot,
    scatter_amp_depth_fr_plot,
    probe_rms_plot,
)
"""
import numpy as np
from pathlib import Path
import pandas as pd
from extraction_barcodes_cl import extract_barcodes

warnings.simplefilter("ignore")
current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(
    0, str(parent_dir) + "\\utilities"
)  ## 0 means search for new dir first and 1 means search for sys.path first
from useful_tools import find_file

# sys.path.insert(0, str(parent_dir) + "\\bonfic")
# from analyse_stimulus_evoked_response import classify_trial_type
n_cpus = os.cpu_count()
n_jobs = n_cpus - 4
global_job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)
#global_job_kwargs = dict(n_jobs=16, chunk_duration="5s", progress_bar=False)
si.set_global_job_kwargs(**global_job_kwargs)
#print(si.get_global_job_kwargs())
# >>> {'n_jobs': 16, 'chunk_duration': '5s', 'progress_bar': False}

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
    oe_folder = Path(thisDir)
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    this_sorter = analysis_methods.get("sorter_name")
    this_experimenter = analysis_methods.get("experimenter")
    if this_sorter.lower() == "spykingcircus2":
        sorter_suffix = "_SC2"
    elif this_sorter.lower() == "kilosort3":
        sorter_suffix = "_KS3"
    elif this_sorter.lower() == "kilosort4":
        sorter_suffix = "_KS4"
    phy_folder_name = "phy" + sorter_suffix

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
            print("load raw stimulus information")
            trial_ext = "trial*.csv"
            this_csv = find_file(stim_directory, trial_ext)
            stim_directory = pd.read_csv(this_csv)
            num_stim = int(stim_directory.shape[0] / 2)
        else:
            stimulus_meta_info = pd.read_pickle(stimulus_meta_file)
            num_stim = stimulus_meta_info.shape[0]
        ##load stimulus meta info

        isi_on_oe = recording.events.timestamp[
            (recording.events.line == 1) & (recording.events.state == 1)
        ]
        stim_on_oe = recording.events.timestamp[
            (recording.events.line == 1) & (recording.events.state == 0)
        ]

        if analysis_methods.get("analyse_stim_evoked_activity") == True:
            if len(stim_on_oe) > num_stim:
                stim_events_times = stim_on_oe[
                    1:
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
                stimulus_meta_info, turning_trials_threshold, walking_trials_threshold
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
                    walk_events_start[1, :] + time_window_behaviours[0] * camera_fps,
                    walk_events_start[1, :] + time_window_behaviours[1] * camera_fps,
                ]
            ).T
            walk_events_end_tw = np.array(
                [
                    walk_events_end[1, :] + time_window_behaviours[0] * camera_fps,
                    walk_events_end[1, :] + time_window_behaviours[1] * camera_fps,
                ]
            ).T
            # fig, axs = plt.subplots(2, 2, figsize=(6, 6))

            # for i, ax in enumerate(axs.flatten()):
            #     file = np.load(adc_folder / os.listdir(adc_folder)[i])
            #     ax.plot(range(len(file)), file)
            # plt.tight_layout()
            # plt.show()
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
    # colormap_name = "coolwarm"
    # COL = MplColorHelper(colormap_name, 0, 8)
    # sm = cm.ScalarMappable(cmap=colormap_name)
    # [exp_date, exp_hour] = csv_file_directory.stem.split("events")[1].split("T")
    # exp_place = csv_file_directory.parts[3]

    # df = pd.read_pickle(tracking_file)
    # if analysis_methods.get("filtering_method") == "sg_filter":
    #     x_all = savgol_filter(df.loc[:, ["intergrated x position"]], 71, 3, axis=0)
    #     y_all = savgol_filter(df.loc[:, ["intergrated y position"]], 71, 3, axis=0)
    # else:
    #     x_all = df.loc[:, ["intergrated x position"]]
    #     y_all = df.loc[:, ["intergrated y position"]]
    # ## here write an additional function to classify stationary, moving and total travel distance
    # vz = df.loc[:, ["delta rotation vector lab z"]].values
    # travel_distance_fbf = np.sqrt(
    #     np.add(np.square(np.diff(x_all, axis=0)), np.square(np.diff(y_all, axis=0)))
    # )
    # velocity_fbf = travel_distance_fbf * camera_fps
    # putative_walk = classify_walk(
    #     velocity_fbf[:, 0], walk_speed_threshold, int(walk_consecutive_length)
    # )
    # dif_putative_walk = np.diff(putative_walk, axis=0)
    # gap_len = np.where(dif_putative_walk > camera_fps * 1)
    # walk_events_end = putative_walk[gap_len]
    # walk_events_start = np.concatenate(
    #     (
    #         [walk_events_end[0] - gap_len[0][0] - 1],
    #         walk_events_end[1:] - gap_len[0][1:] + gap_len[0][:-1],
    #     )
    # )

    # pcm = plt.pcolormesh(velocity_tbt[24:53], cmap="magma", vmin=0, vmax=100)
    # # for i in walk_events_start:
    # #     plt.axvline(x=i, color="w")
    # plt.show()
    # if analysis_methods.get("plotting_trajectory") == True:
    #     plt.plot(x_all, y_all, c=np.arange(len(y_all)), marker=".", alpha=0.5)
    #     plt.show()

    # sorting_wout_excess_spikes = scur.remove_excess_spikes(
    #     sorting_spikes, recording_saved
    # )
    # sorting_spikes = sorting_wout_excess_spikes
    # if analysis_methods.get("extract_waveform_sparse") == True:
    #     waveform_folder_name = "waveforms_sparse" + sorter_suffix
    #     we = si.extract_waveforms(
    #         recording_saved,
    #         sorting_spikes,
    #         folder=oe_folder / waveform_folder_name,
    #         sparse=True,
    #         overwrite=True,
    #         **job_kwargs,
    #     )
    # else:
    #     waveform_folder_name = "waveforms_dense" + sorter_suffix
    #     we = si.extract_waveforms(
    #         recording_saved,
    #         sorting_spikes,
    #         folder=oe_folder / waveform_folder_name,
    #         sparse=False,
    #         overwrite=True,
    #         **job_kwargs,
    #     )
    #     all_templates = we.get_all_templates()
    #     print(f"All templates shape: {all_templates.shape}")
    #     for unit in sorting_spikes.get_unit_ids()[::10]:
    #         waveforms = we.get_waveforms(unit_id=unit)
    #         spiketrain = sorting_spikes.get_unit_spike_train(unit)
    #         print(
    #             f"Unit {unit} - num waveforms: {waveforms.shape[0]} - num spikes: {len(spiketrain)}"
    #         )

    #     sparsity = si.compute_sparsity(we, method="radius", radius_um=100.0)
    #     #  check the sparsity for some units
    #     for unit_id in sorting_spikes.unit_ids[::30]:
    #         print(unit_id, list(sparsity.unit_id_to_channel_ids[unit_id]))
    #     if analysis_methods.get("extract_waveform_sparse_explicit") == True:
    #         waveform_folder_name = "waveforms_sparse_explicit" + sorter_suffix
    #         we = si.extract_waveforms(
    #             recording_saved,
    #             sorting_spikes,
    #             folder=oe_folder / waveform_folder_name,
    #             sparse=sparsity,
    #             overwrite=True,
    #             **job_kwargs,
    #         )
    #         # the waveforms are now sparse
    #         for unit_id in we.unit_ids[::10]:
    #             waveforms = we.get_waveforms(unit_id=unit_id)
    #             print(unit_id, waveforms.shape)
    # ##evaluating the spike sorting
    # pc = spost.compute_principal_components(
    #     we, n_components=3, load_if_exists=False, **job_kwargs
    # )
    # all_labels, all_pcs = pc.get_all_projections()
    # print(f"All PC scores shape: {all_pcs.shape}")
    # we.get_available_extension_names()
    # pc = we.load_extension("principal_components")
    # all_labels, all_pcs = pc.get_data()
    # print(all_pcs.shape)
    # amplitudes = spost.compute_spike_amplitudes(
    #     we, outputs="by_unit", load_if_exists=True, **job_kwargs
    # )
    # unit_locations = spost.compute_unit_locations(
    #     we, method="monopolar_triangulation", load_if_exists=True
    # )
    # spike_locations = spost.compute_spike_locations(
    #     we, method="center_of_mass", load_if_exists=True, **job_kwargs
    # )
    # # spike_clusters=find_cluster_from_peaks(recording_saved, peaks, method='stupid', method_kwargs={}, extra_outputs=False, **job_kwargs)
    # similarity = spost.compute_template_similarity(we)
    # template_metrics = spost.compute_template_metrics(we)
    # qm_params = sq.get_default_qm_params()
    # metric_names = sq.get_quality_metric_list()
    sorting_analyzer=si.load_sorting_analyzer(folder=oe_folder/"sorting_analyzer")
    print(sorting_analyzer.get_loaded_extension_names())
    if (
        analysis_methods.get("load_curated_spikes") == True
        and (oe_folder / phy_folder_name).is_dir()
    ):
        sorting_spikes = se.read_phy(
            oe_folder / phy_folder_name, exclude_cluster_groups=["noise"]
        )
        if (oe_folder / "preprocessed_compressed.zarr").is_dir():
            recording_saved = si.read_zarr(oe_folder / "preprocessed_compressed.zarr")
            print(recording_saved.get_property_keys())
        elif (oe_folder / "preprocessed").is_dir():
            recording_saved = si.load_extractor(oe_folder / "preprocessed")
        else:
            print(f"no pre-processed folder found. Unable to extract waveform")
            return sorting_spikes

        ###start to analyse spikes or loading info from sorted spikes
        recording_saved.annotate(is_filtered=True)
        # spikeinterface.SortingAnalyzer

        sorting_analyzer = si.create_sorting_analyzer(
            sorting=sorting_spikes,
            recording=recording_saved,
            sparse=True,  # default
            format="memory",  # default
        )

        sorting_analyzer.compute(["random_spikes","waveforms","templates","noise_levels","spike_amplitudes", "spike_locations", "unit_locations"])
        sorting_analyzer.compute(['correlograms','template_similarity'])
        '''
        compute_dict = {
    'principal_components': {'n_components': 3, 'mode': 'by_channel_local'},
    'templates': {'operators': ["average"]}}
        sorting_analyzer.compute(compute_dict)
        sorting_analyzer = si.create_sorting_analyzer(
            sorting=sorting_spikes,
            recording=recording_saved,
            sparse=True,  # default
            format="binary_folder",
            folder=oe_folder/"sorting_analyzer" # default
        )
        compute_dict = {
    'unit_locations': {'method':["monopolar_triangulation"]}}
    sorting_analyzer.compute(compute_dict)
        sorting_analyzer.compute(input="spike_locations",
                         ms_before=0.5,
                         ms_after=0.5,
                         spike_retriever_kwargs=dict(
                            channel_from_template=True,
                            radius_um=50,
                            peak_sign="neg"
                                          ),
                         method="center_of_mass")
        sorting_analyzer.compute(["random_spikes","waveforms","templates","noise_levels","spike_amplitudes",'correlograms','template_similarity'],save=True)
        import numcodecs
        sorting_analyzer_zarr=sorting_analyzer.save_as(folder=oe_folder/"sorting_analyzer.zarr",format="zarr")
        '''        

        
        
        drift_ptps, drift_stds, drift_mads = sqm.compute_drift_metrics(
            sorting_analyzer=sorting_analyzer)
        # plot_sorting_summary(sorting_analyzer,curation=True,backend='sortingview') use this in jupyter notebook
        sorting_analyzer.get_loaded_extension_names()
        spike_time_list = []
        spike_amp_list = []
        cluster_id_list = []
        for unit in sorting_spikes.get_unit_ids():
            print(
                f"with {this_sorter} sorter, Spike train of a unit:{sorting_spikes.get_unit_spike_train(unit_id=unit)}"
            )
            spike_times = sorting_spikes.get_unit_spike_train(unit_id=unit) / float(
                sorting_spikes.sampling_frequency
            )
            spike_time_list.append(spike_times)
            cluster_id_list.append(np.ones(len(spike_times), dtype=int) * unit)
            # df.to_csv('example.tsv', sep="\t")
            # np.savetxt("data3.tsv", a,  delimiter = "\t")
            # spike_amp_list.append(amplitudes[0][unit])
            # spike_amps = amplitudes[0][unit]
        spike_time_all = np.concatenate(spike_time_list)
        cluster_id_all = np.concatenate(cluster_id_list)
        spike_time_all_sorted, cluster_id_all_sorted = sort_arrays(
            spike_time_all, cluster_id_all
        )
        spike_count, cluster_id = get_spike_counts_in_bins(
            spike_time_all_sorted, cluster_id_all_sorted, stim_events_tw
        )
        scatter_amp_depth_fr_plot(
            spike_amps=0,
            spike_clusters=cluster_id_all,
            spike_depths=0,
            spike_times=spike_time_all,
        )
        # Compute rate (for all clusters of interest)
        num_trial = stim_events_tw.shape[0]
        num_neuron = len(np.unique(cluster_id_all))
        spike_rate = np.zeros((num_neuron, num_trial))
        spike_rate = spike_count / (time_window[1] - time_window[0])
        # this_event>sorting_spikes.get_unit_spike_train(unit_id=unit)/float(sorting_spikes.sampling_frequency)
        # unique, counts = np.unique(cluster_id_all, return_counts=True)#check unique spike counts
        if analysis_methods.get("analysis_by_stimulus_type") == True:
            stim_type = analysis_methods.get("stim_type")
            for this_id in np.unique(cluster_id_all):
                # for thisStim in stim_type:

                #     # troubleshoot this part. Dont know why it exceeds the max size
                #     this_event = stim_events_times[
                #         stimulus_meta_info.loc[:, "stim_type"] > thisStim
                #     ]
                # peths, binned_spikes = singlecell.calculate_peths(
                #     spike_time_all,
                #     cluster_id_all,
                #     [this_id],
                #     stim_events_times,
                #     pre_time=10,
                #     post_time=20,
                #     bin_size=0.025,
                #     smoothing=0.025,
                #     return_fr=True,
                # )
                # walk_events_start_oe time points when walk starts
                peri_event_time_histogram(
                    spike_time_all,
                    cluster_id_all,
                    event_of_interest,
                    this_id,
                    t_before=abs(time_window_behaviours[0]),
                    t_after=time_window_behaviours[1],
                    bin_size=0.025,
                    smoothing=0.025,
                    include_raster=True,
                    raster_kwargs={"color": "black", "lw": 1},
                )
            # testDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23012\231126\coherence\session1"
            # pd_ext = "stimulus_meta_info.pickle"
            # this_PD = find_file(testDir, pd_ext)
            # stimulus_meta_info = pd.read_pickle(this_PD)
        return spike_count, cluster_id


if __name__ == "__main__":
    # thisDir = r"C:\Users\neuroLaptop\Documents\Open Ephys\P-series-32channels\GN00003\2023-12-28_14-39-40"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN2300x\231123\coherence\2024-05-05_22-57-50"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23019\240507\coherence\session1\2024-05-07_23-08-55"
    thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23018\240422\coherence\session2\2024-04-22_01-09-50"
    # thisDir = r"C:\Users\neuroPC\Documents\Open Ephys\2024-02-01_15-25-25"
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    align_async_signals(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
