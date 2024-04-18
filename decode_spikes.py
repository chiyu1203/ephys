import time, os, json, warnings, sys
import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.sorters as ss
import spikeinterface.qualitymetrics as sq
import spikeinterface.exporters as sep

# import spikeinterface.comparison as sc
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.trials import get_event_aligned_raster, get_psth
from brainbox.ephys_plots import scatter_raster_plot
import numpy as np
from pathlib import Path
import probeinterface as pi
from probeinterface.plotting import plot_probe
import numcodecs
import pandas as pd

warnings.simplefilter("ignore")
import spikeinterface.curation as scur

sys.path.insert(1, r"C:\Users\neuroPC\Documents\GitHub\bonfic")
from align_data_with_ttl import find_file


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


def main(thisDir, json_file):
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
    result_folder_name = "results" + sorter_suffix
    sorting_folder_name = "sorting" + sorter_suffix
    phy_folder_name = "phy" + sorter_suffix
    n_cpus = os.cpu_count()
    n_jobs = n_cpus - 4
    job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)

    if analysis_methods.get("load_curated_spikes") == True:
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
        recording_saved.annotate(is_filtered=True)
    if analysis_methods.get("aligning_with_stimuli") == True:
        full_raw_rec = se.read_openephys(oe_folder, load_sync_timestamps=True)
        aux_events = se.read_openephys_event(oe_folder)
        events_times = aux_events.get_event_times(
            channel_id=aux_events.channel_ids[1], segment_index=0
        )  # this record ON phase of sync pulse
        time_window = np.array([-0.1, 0.0])
        events_tw = np.array(
            [events_times[1:] + time_window[0], events_times[1:] + time_window[1]]
        ).T  # ignore the first event, which is recorded when pressing the start delivering stimulus button on Bonsai

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
        # spike_amp_list.append(amplitudes[0][unit])
        # spike_amps = amplitudes[0][unit]
    spike_time_all = np.concatenate(spike_time_list)
    cluster_id_all = np.concatenate(cluster_id_list)
    spike_time_all_sorted, cluster_id_all_sorted = sort_arrays(
        spike_time_all, cluster_id_all
    )
    spike_count, cluster_id = get_spike_counts_in_bins(
        spike_time_all_sorted, cluster_id_all_sorted, events_tw
    )

    # Compute rate (for all clusters of interest)
    num_trial = events_tw.shape[0]
    num_neuron = len(np.unique(cluster_id_all))
    spike_rate = np.zeros((num_neuron, num_trial))
    spike_rate = spike_count / (time_window[1] - time_window[0])
    # this_event>sorting_spikes.get_unit_spike_train(unit_id=unit)/float(sorting_spikes.sampling_frequency)

    if analysis_methods.get("analysis_by_stimulus_type") == True:
        testDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23012\231126\coherence\session1"
        pd_pattern = "behavioural_summary.pickle"
        this_PD = find_file(testDir, pd_pattern)
        behavioural_summary = pd.read_pickle(this_PD)


if __name__ == "__main__":
    # thisDir = r"C:\Users\neuroLaptop\Documents\Open Ephys\P-series-32channels\GN00003\2023-12-28_14-39-40"
    thisDir = r"Z:\DATA\experiment_openEphys\P-series-32channels\2024-02-01_15-25-25"
    # thisDir = r"C:\Users\neuroPC\Documents\Open Ephys\2024-02-01_15-25-25"
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    main(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
