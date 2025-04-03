import time, os, json, warnings
import spikeinterface.core as si
import probeinterface as pi
import spikeinterface.preprocessing as spre
from pathlib import Path
from raw2si import generate_sorter_suffix
import spikeinterface.curation as scur
import spikeinterface.exporters as sep
import spikeinterface.qualitymetrics as sqm
import spikeinterface.extractors as se
import numpy as np
import matplotlib.pyplot as plt
from spikeinterface.widgets import plot_sorting_summary
from brainbox.plot import peri_event_time_histogram, driftmap_color, driftmap
import pandas as pd
import spikeinterface.widgets as sw
import matplotlib as mpl
from matplotlib import cm

warnings.simplefilter("ignore")
n_cpus = os.cpu_count()
n_jobs = n_cpus - 4
global_job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)
# global_job_kwargs = dict(n_jobs=16, chunk_duration="5s", progress_bar=False)
si.set_global_job_kwargs(**global_job_kwargs)
"""
This pipeline uses spikeinterface as a backbone. This file includes extracting waveform, calculating quality metrics and exporting to phy, doing analysis on putative spikes
"""


class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


colormap_name = "coolwarm"
COL = MplColorHelper(colormap_name, 0, 8)


def spike_overview(
    oe_folder,
    this_sorter,
    sorting_spikes,
    sorting_analyzer,
    recording_saved,
    unit_labels,
    merge_similiar_unit_for_overview=False,
):
    # load analysed spike data
    ext = sorting_analyzer.get_extension("spike_locations")
    spike_loc = ext.get_data(outputs="by_unit")
    ext = sorting_analyzer.get_extension("unit_locations")
    unit_loc = ext.get_data(outputs="by_unit")
    ext = sorting_analyzer.get_extension("spike_amplitudes")
    spike_amp = ext.get_data(outputs="by_unit")
    fraction_missing = sqm.compute_amplitude_cutoffs(
        sorting_analyzer=sorting_analyzer, peak_sign="neg"
    )
    print(
        f"fraction missing: {fraction_missing} in these units, meaning the fraction of false negatives (missed spikes) by the sorter"
    )

    # spost.align_sorting(sorting_analyzer, drift_ptps)
    # si.get_template_extremum_channel_peak_shift
    ax = sw.plot_unit_templates(sorting_analyzer, backend="matplotlib")
    fig_name = f"preview_unit_template.png"
    fig_dir = oe_folder / fig_name
    ax.figure.savefig(fig_dir)
    spike_time_list = []
    spike_amp_list = []
    cluster_id_list = []
    spike_loc_list = []
    unit_loc_list = []
    i = 0
    for this_unit, this_label in zip(sorting_spikes.get_unit_ids(), unit_labels):
        print(
            f"with {this_sorter} sorter, Spike train of a unit:{sorting_spikes.get_unit_spike_train(unit_id=this_unit)}"
        )
        spike_times = sorting_spikes.get_unit_spike_train(unit_id=this_unit) / float(
            sorting_spikes.sampling_frequency
        )
        spike_time_list.append(spike_times)
        if merge_similiar_unit_for_overview == True:
            print(
                "a rough analysis that assign potentially same units into new ids. This method is not back by drifting matrics so should be only used for a quick look"
            )
            if this_label == "good":
                this_cluster_id = np.ones(len(spike_times), dtype=int) * 1000
            elif this_label == "mua":
                this_cluster_id = np.ones(len(spike_times), dtype=int) * 2000
            else:
                print(
                    f"{this_label} not found. Please check if this is the kind of units you want to select"
                )
                continue
        else:
            this_cluster_id = np.ones(len(spike_times), dtype=int) * this_unit
        cluster_id_list.append(this_cluster_id)
        spike_amp_list.append(spike_amp[0][this_unit])
        # dont know what is the best way to convert tuple so convert it into pandas dataframe first and then extract np.array from dataframe
        # maybe list comprehension is faster
        # access tuple: spike_loc[0][12][0] access array element: spike_loc[0][12][0][0] spike_loc[0][12][0][1]

        spike_loc_df = pd.DataFrame(spike_loc[0][this_unit], columns=["x", "y"])
        spike_loc_df["distance"] = np.sqrt(
            spike_loc_df["x"] ** 2 + spike_loc_df["y"] ** 2
        )
        spike_loc_list.append(spike_loc_df["distance"].values)

        this_unit_loc = np.sqrt(
            unit_loc[this_unit][0] ** 2 + unit_loc[this_unit][1] ** 2
        )
        unit_loc_list.append(this_unit_loc)

        plt.figure()
        ax = plt.gca()
        ax.scatter(x=spike_times, y=spike_loc_df["distance"].values, c=COL.get_rgb(i))
        ax.set_xlim([0, recording_saved.get_total_duration()])
        ax.set_ylim([250, 350])
        fig_name = f"spike_location_unit{this_unit}.svg"
        fig_dir = oe_folder / fig_name
        ax.figure.savefig(fig_dir)
        # plot drift map based on brainbox's plotting function
        ax_drift, x_lim, y_lim = driftmap_color(
            clusters_depths=this_unit_loc,
            spikes_times=spike_times,
            spikes_amps=spike_amp[0][this_unit],
            spikes_depths=spike_loc_df["distance"].values,
            spikes_clusters=this_cluster_id,
            ax=None,
            axesoff=False,
            return_lims=True,
        )
        # ax_drift.set_xlim([0, recording_saved.get_total_duration()])
        # ax_drift.set_ylim([250, 350])
        ax_drift.set_xlim(x_lim)
        ax_drift.set_ylim(y_lim)
        fig_name = f"driftmap_unit{this_unit}.svg"
        fig_dir = oe_folder / fig_name
        ax_drift.figure.savefig(fig_dir)
        i += 1

    ##try to use this driftmap_color or driftmap
    return (
        np.concatenate(spike_time_list),
        np.concatenate(cluster_id_list),
        np.concatenate(spike_amp_list),
        np.concatenate(spike_loc_list),
    )


def get_preprocessed_recording(oe_folder):
    if (oe_folder / "preprocessed_compressed.zarr").is_dir():
        recording_saved = si.read_zarr(oe_folder / "preprocessed_compressed.zarr")
        print(recording_saved.get_property_keys())
    elif (oe_folder / "preprocessed").is_dir():
        recording_saved = si.load_extractor(oe_folder / "preprocessed")
    else:
        print(f"no pre-processed folder found. Unable to extract waveform")
        print("create temporary option to re-run the pre-processing script to generate the recording object")
        raw_rec = se.read_openephys(oe_folder, load_sync_timestamps=True)
        stacked_probes = pi.read_probeinterface("H10_stacked_probes.json")
        probe = stacked_probes.probes[0]
        raw_rec = raw_rec.set_probe(probe,group_mode='by_shank')
        recording_f = spre.bandpass_filter(raw_rec, freq_min=600, freq_max=6000,dtype="float32")
        recordings_dict = recording_f.split_by(property='group', outputs='dict')
        recording_cmr = spre.common_reference(
                recordings_dict, reference="global", operator="median"
            )
        recording_saved=si.aggregate_channels(recording_cmr)
        #return None
    recording_saved.annotate(
        is_filtered=True
    )  # note down this recording is bandpass filtered and cmr
    return recording_saved


def calculate_analyzer_extension(sorting_analyzer):
    sorting_analyzer.compute(
        ["random_spikes", "isi_histograms", "correlograms", "noise_levels"]
    )
    # sorting_analyzer.compute(["waveforms", "principal_components", "templates"])
    sorting_analyzer.compute("waveforms")
    compute_dict = {
        "principal_components": {"n_components": 3, "mode": "by_channel_local"},
        "templates": {"operators": ["average"]},
    }
    sorting_analyzer.compute(compute_dict)
    compute_dict = {
        "unit_locations": {"method": "monopolar_triangulation"},
        "spike_amplitudes": {"peak_sign": "neg"},
    }
    sorting_analyzer.compute(compute_dict)
    #isolate analysis of spike location here as it has more parameters
    # sorting_analyzer.compute(input="spike_locations")
    # drift_ptps, drift_stds, drift_mads = sqm.compute_drift_metrics(sorting_analyzer=sorting_analyzer,peak_sign="neg")
    # sorting_analyzer.compute(
    # input="spike_locations",
    # ms_before=0.5,
    # ms_after=0.5,
    # spike_retriever_kwargs=dict(
    #     channel_from_template=True,
    #     radius_um=50,
    #     peak_sign="neg"
    # ),
    # method="center_of_mass")
    sorting_analyzer.compute(
        [
            "spike_locations",
            "amplitude_scalings",
            "template_metrics",
            "template_similarity",
            "quality_metrics",
        ]
    )


def si2phy(thisDir, json_file):
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
    remove_excess_spikes = 1

    if (
        analysis_methods.get("load_analyser_from_disc") == True
        and (oe_folder / analyser_folder_name).is_dir()
    ):
        sorting_analyzer = si.load_sorting_analyzer(
            folder=oe_folder / analyser_folder_name
        )
        print("load_the existing analyser to do more computation or plot summary")

    elif (
        analysis_methods.get("load_curated_spikes") == True
        and (oe_folder / phy_folder_name).is_dir()
    ):
        print("overwrite the existing sorting_analyser with phy-curated spikes")
        if analysis_methods.get("include_MUA") == True:
            cluster_group_interest = ["noise"]
        else:
            cluster_group_interest = ["noise", "mua"]
        sorting_spikes = se.read_phy(
            oe_folder / phy_folder_name, exclude_cluster_groups=cluster_group_interest
        )
        sorting_duduplicated = scur.remove_duplicated_spikes(sorting_spikes)
        sorting_spikes = sorting_duduplicated
        unit_labels = sorting_spikes.get_property("quality")
        recording_saved = get_preprocessed_recording(oe_folder)
        sorting_analyzer = si.create_sorting_analyzer(
            sorting=sorting_spikes,
            recording=recording_saved,
            sparse=True,  # default
            format="binary_folder",
            folder=oe_folder / analyser_folder_name,
            overwrite=True,  # default  # default
        )
        calculate_analyzer_extension(sorting_analyzer)
        _, _, _, _ = spike_overview(
            oe_folder,
            this_sorter,
            sorting_spikes,
            sorting_analyzer,
            recording_saved,
            unit_labels,
        )

    elif (
        oe_folder / sorting_folder_name
    ).is_dir():  # need to double check the difference between sorting_folder_name and result_folder_name
        print(
            "create a sorting_analyser with fresh sorted spikes from automatic sorters"
        )
        sorting_spikes = si.load_extractor(
            oe_folder / sorting_folder_name
        )  # this acts quite similar than above one line.
        recording_saved = get_preprocessed_recording(oe_folder)
        if remove_excess_spikes:
            sorting_wout_excess_spikes = scur.remove_excess_spikes(
                sorting_spikes, recording_saved
            )
            sorting_spikes = sorting_wout_excess_spikes
        sorting_analyzer = si.create_sorting_analyzer(
            sorting=sorting_spikes,
            recording=recording_saved,
            sparse=True,  # default
            format="memory",  # default
        )
        calculate_analyzer_extension(sorting_analyzer)
        if (
            analysis_methods.get("export_to_phy") == True
            and analysis_methods.get("overwrite_existing_phy") == True
        ):
            sep.export_to_phy(
                sorting_analyzer,
                output_folder=oe_folder / phy_folder_name,
                compute_amplitudes=True,
                compute_pc_features=True,
                copy_binary=True,
                remove_if_exists=True,
            )
        else:
            isi_viol_thresh = 0.5
            amp_cutoff_thresh = 0.1
            our_query = f"amplitude_cutoff < {amp_cutoff_thresh} & isi_violations_ratio < {isi_viol_thresh}"
            print(our_query)
            keep_units = sqm.query(our_query)
            keep_unit_ids = keep_units.index.values
    else:
        return print(
            f"{sorting_folder_name} is not found. Noting can be done here without some putative spikes..."
        )

    ax = sw.plot_unit_templates(sorting_analyzer, backend="matplotlib")
    fig_name = f"preview_unit_template.png"
    fig_dir = oe_folder / fig_name
    ax.figure.savefig(fig_dir)

    # drift_ptps, drift_stds, drift_mads = sqm.compute_drift_metrics(
    #     sorting_analyzer=sorting_analyzer
    # )
    if analysis_methods.get("export_report") == True:
        sep.export_report(
            sorting_analyzer, output_folder=oe_folder / report_folder_name
        )


if __name__ == "__main__":
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23019\240507\coherence\session1\2024-05-07_23-08-55"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23018\240422\coherence\session2\2024-04-22_01-09-50"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23015\240201\coherence\session1\2024-02-01_15-25-25"
    #thisDir = r"Z:\DATA\experiment_openEphys\P-series-32channels\2025-02-26_17-00-43"
    thisDir = r"Z:\DATA\experiment_openEphys\H-series-128channels\2025-03-23_21-33-38"
    # thisDir = r"C:\Users\neuroPC\Documents\Open Ephys\2024-02-01_15-25-25"
    #thisDir = r"D:\Open Ephys\2025-02-23_20-39-04"
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    si2phy(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
