import time, os, json, warnings
import spikeinterface.core as si
from pathlib import Path
from raw2si import generate_sorter_suffix
import spikeinterface.curation as scur
import spikeinterface.postprocessing as spost
import spikeinterface.exporters as sep
import spikeinterface.qualitymetrics as sq

warnings.simplefilter("ignore")
"""
This pipeline uses spikeinterface as a backbone. This file includes extracting waveform, calculating quality metrics and exporting to phy, doing analysis on putative spikes
"""


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
    phy_folder_name = phy_folder_name = "phy" + sorter_suffix
    n_cpus = os.cpu_count()
    n_jobs = n_cpus - 4
    job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)
    remove_excess_spikes = 0

    ##extracting waveform
    # the extracted waveform based on sparser signals (channels) makes the extraction faster.
    # However, if the channels are not dense enough the right waveform can not be properly extracted.
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

    if remove_excess_spikes:
        sorting_wout_excess_spikes = scur.remove_excess_spikes(
            sorting_spikes, recording_saved
        )
        sorting_spikes = sorting_wout_excess_spikes
    if analysis_methods.get("extract_waveform_sparse") == True:
        waveform_folder_name = "waveforms_sparse" + sorter_suffix
        we = si.extract_waveforms(
            recording_saved,
            sorting_spikes,
            folder=oe_folder / waveform_folder_name,
            sparse=True,
            overwrite=True,
            **job_kwargs,
        )
    else:
        waveform_folder_name = "waveforms_dense" + sorter_suffix
        we = si.extract_waveforms(
            recording_saved,
            sorting_spikes,
            folder=oe_folder / waveform_folder_name,
            sparse=False,
            overwrite=True,
            **job_kwargs,
        )
        all_templates = we.get_all_templates()
        print(f"All templates shape: {all_templates.shape}")
        for unit in sorting_spikes.get_unit_ids()[::10]:
            waveforms = we.get_waveforms(unit_id=unit)
            spiketrain = sorting_spikes.get_unit_spike_train(unit)
            print(
                f"Unit {unit} - num waveforms: {waveforms.shape[0]} - num spikes: {len(spiketrain)}"
            )

        sparsity = si.compute_sparsity(we, method="radius", radius_um=100.0)
        #  check the sparsity for some units
        for unit_id in sorting_spikes.unit_ids[::30]:
            print(unit_id, list(sparsity.unit_id_to_channel_ids[unit_id]))
        if analysis_methods.get("extract_waveform_sparse_explicit") == True:
            waveform_folder_name = "waveforms_sparse_explicit" + sorter_suffix
            we = si.extract_waveforms(
                recording_saved,
                sorting_spikes,
                folder=oe_folder / waveform_folder_name,
                sparse=sparsity,
                overwrite=True,
                **job_kwargs,
            )
            # the waveforms are now sparse
            for unit_id in we.unit_ids[::10]:
                waveforms = we.get_waveforms(unit_id=unit_id)
                print(unit_id, waveforms.shape)
    ##evaluating the spike sorting
    pc = spost.compute_principal_components(
        we, n_components=3, load_if_exists=False, **job_kwargs
    )
    all_labels, all_pcs = pc.get_all_projections()
    print(f"All PC scores shape: {all_pcs.shape}")
    we.get_available_extension_names()
    pc = we.load_extension("principal_components")
    all_labels, all_pcs = pc.get_data()
    print(all_pcs.shape)
    amplitudes = spost.compute_spike_amplitudes(
        we, outputs="by_unit", load_if_exists=True, **job_kwargs
    )
    unit_locations = spost.compute_unit_locations(
        we, method="monopolar_triangulation", load_if_exists=True
    )
    spike_locations = spost.compute_spike_locations(
        we, method="center_of_mass", load_if_exists=True, **job_kwargs
    )
    ##spike_clusters=find_cluster_from_peaks(recording_saved, peaks, method='stupid', method_kwargs={}, extra_outputs=False, **job_kwargs)
    ccgs, bins = spost.compute_correlograms(we)
    similarity = spost.compute_template_similarity(we)
    template_metrics = spost.compute_template_metrics(we)
    qm_params = sq.get_default_qm_params()
    metric_names = sq.get_quality_metric_list()
    if we.return_scaled:
        qm = sq.compute_quality_metrics(
            we,
            metric_names=metric_names,
            verbose=True,
            qm_params=qm_params,
            **job_kwargs,
        )
    print(we.get_available_extension_names())  # check available extension

    ##curation
    # the safest way to curate spikes are manual curation, which phy seems to be a good package to deal with that
    # When exporting spikes data to phy, amplitudes and pc features can also be calculated
    # if you do not wish to use phy, we can calculate quality metrics with other packages in spikeinterface.
    ##exporting to phy
    ##still need to check whether methods are used to compute pc features and amplitude when exporting to phy, and whether
    ## we want those methods to be default methods.
    ## If not, we should get some spost methods before this step and turn the two computer options in export_to_phy off.
    if (
        analysis_methods.get("export_to_phy") == True
        and analysis_methods.get("overwrite_existing_phy") == True
    ):
        phy_folder_name = "phy" + sorter_suffix
        sep.export_to_phy(
            we,
            output_folder=oe_folder / phy_folder_name,
            compute_amplitudes=True,
            compute_pc_features=True,
            copy_binary=True,
            remove_if_exists=True,
            **job_kwargs,
        )
    else:
        qm = sq.compute_quality_metrics(
            we,
            metric_names=metric_names,
            verbose=True,
            qm_params=qm_params,
            **job_kwargs,
        )
        print(qm)
        # sq.plot_quality_metrics(we, include_metrics=["amplitude_cutoff", "presence_ratio", "isi_violations_ratio", "snr"],
        #                 backend="ipywidgets")
        isi_viol_thresh = 0.5
        amp_cutoff_thresh = 0.1
        our_query = f"amplitude_cutoff < {amp_cutoff_thresh} & isi_violations_ratio < {isi_viol_thresh}"
        print(our_query)
        keep_units = qm.query(our_query)
        keep_unit_ids = keep_units.index.values
        we = we.select_units(keep_unit_ids, new_folder=oe_folder / "waveforms_curated")

    ##outputing a report
    if analysis_methods.get("export_report") == True:
        report_folder_name = "report" + sorter_suffix
        sep.export_report(we, output_folder=oe_folder / report_folder_name)
    return sorting_spikes, we


if __name__ == "__main__":
    # thisDir = r"C:\Users\neuroLaptop\Documents\Open Ephys\P-series-32channels\GN00003\2023-12-28_14-39-40"
    thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN2300x\231123\coherence\2024-05-05_22-57-50"
    thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23019\240507\coherence\session1\2024-05-07_23-08-55"
    # thisDir = r"C:\Users\neuroPC\Documents\Open Ephys\2024-02-01_15-25-25"
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    si2phy(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
