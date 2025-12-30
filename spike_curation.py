import time, os, json, warnings,sys
import spikeinterface.core as si
import probeinterface as pi
import spikeinterface.preprocessing as spre
from pathlib import Path
from raw2si import generate_sorter_suffix,get_preprocessed_recording,motion_correction_shankbyshank
import spikeinterface.curation as scur
import spikeinterface.exporters as sep
import spikeinterface.qualitymetrics as sqm
import spikeinterface.extractors as se
import numpy as np
import matplotlib.pyplot as plt
from spikeinterface.widgets import plot_sorting_summary
from estimate_drift_motion import AP_band_drift_estimation, LFP_band_drift_estimation
from brainbox.plot import peri_event_time_histogram, driftmap_color, driftmap
import pandas as pd
import spikeinterface.widgets as sw
import matplotlib as mpl
from matplotlib import cm
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

warnings.simplefilter("ignore")
n_cpus = os.cpu_count()
n_jobs = n_cpus - 4

global_job_kwargs = dict(n_jobs=n_jobs, chunk_duration="2s", progress_bar=True)
# global_job_kwargs = dict(n_jobs=16, chunk_duration="5s", progress_bar=False)
si.set_global_job_kwargs(**global_job_kwargs)
current_working_directory = Path.cwd()
parent_dir = current_working_directory.resolve().parents[0]
sys.path.insert(
    0, str(parent_dir) + "\\utilities"
)  ## 0 means search for new dir first and 1 means search for sys.path first
from useful_tools import find_file
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

def calculate_moving_avg(label_df, confidence_label, window_size):

    label_df[f'{confidence_label}_decile'] = pd.cut(label_df[confidence_label], 10, labels=False, duplicates='drop')
    # Group by decile and calculate the proportion of correct labels (agreement)
    p_label_grouped = label_df.groupby(f'{confidence_label}_decile')['model_x_human_agreement'].mean()
    # Convert decile to range 0-1
    p_label_grouped.index = p_label_grouped.index / 10
    # Sort the DataFrame by confidence scores
    label_df_sorted = label_df.sort_values(by=confidence_label)

    p_label_moving_avg = label_df_sorted['model_x_human_agreement'].rolling(window=window_size).mean()

    return label_df_sorted[confidence_label], p_label_moving_avg

def spike_overview(
    oe_folder,
    this_sorter,
    sorting_spikes,
    sorting_analyzer,
    recording_for_analysis,
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
    plot_fig=False
    if plot_fig:
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
        if plot_fig:
            plt.figure()
            ax = plt.gca()
            ax.scatter(x=spike_times, y=spike_loc_df["distance"].values, c=COL.get_rgb(i))
            ax.set_xlim([0, recording_for_analysis.get_total_duration()])
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
            # ax_drift.set_xlim([0, recording_for_analysis.get_total_duration()])
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
def calculate_analyzer_extension(sorting_analyzer):
    # ext = sorting_analyzer.get_extension("spike_amplitudes")
    # ext_data = ext.get_data()
    ##use this to get documentation ext.get_data?

    ''' 
    Question: PCA why 5 instead of 3; templates why operator average std and not ms before and after; spike_location why center of_mass
    print(sorting_analyzer.get_default_extension_params('random_spikes'))
    {'method': 'uniform', 'max_spikes_per_unit': 500, 'margin_size': None, 'seed': None}
    print(sorting_analyzer.get_default_extension_params('noise_levels'))
    {}
    print(sorting_analyzer.get_default_extension_params('correlograms'))
    {'window_ms': 50.0, 'bin_ms': 1.0, 'method': 'auto'}
    print(sorting_analyzer.get_default_extension_params('isi_histograms'))
    {'window_ms': 50.0, 'bin_ms': 1.0, 'method': 'auto'}
    print(sorting_analyzer.get_default_extension_params('templates'))
    {'ms_before': 1.0, 'ms_after': 2.0, 'operators': None}
    print(sorting_analyzer.get_default_extension_params('waveforms'))
    {'ms_before': 1.0, 'ms_after': 2.0, 'dtype': None}
    print(sorting_analyzer.get_default_extension_params('principal_components'))
    {'n_components': 5, 'mode': 'by_channel_local', 'whiten': True, 'dtype': 'float32'}
    print(sorting_analyzer.get_default_extension_params('unit_locations'))
    {'method': 'monopolar_triangulation'}
    print(sorting_analyzer.get_default_extension_params('spike_amplitudes'))
    {'peak_sign': 'neg'}
    print(sorting_analyzer.get_default_extension_params('spike_locations'))
    {'ms_before': 0.5, 'ms_after': 0.5, 'spike_retriver_kwargs': None, 'method': 'center_of_mass', 'method_kwargs': {}}
    print(sorting_analyzer.get_default_extension_params('template_metrics'))
    {'metric_names': None, 'peak_sign': 'neg', 'upsampling_factor': 10, 'sparsity': None, 'metric_params': None, 'metrics_kwargs': None, 'include_multi_channel_metrics': False, 'delete_existing_metrics': False}
    print(sorting_analyzer.get_default_extension_params('template_similarity'))
    {'method': 'cosine', 'max_lag_ms': 0, 'support': 'union'}
    print(sorting_analyzer.get_default_extension_params('quality_metrics'))
    {'metric_names': None, 'metric_params': None, 'qm_params': None, 'peak_sign': None, 'seed': None, 'skip_pc_metrics': False, 'delete_existing_metrics': False, 'metrics_to_compute': None}

    '''
    sorting_analyzer.compute(
        {
        "noise_levels":{},
        "random_spikes":{'max_spikes_per_unit':1000},
        "correlograms":{'bin_ms':0.5},
        "isi_histograms":{},
        "waveforms":{},
        "principal_components":{},
        "templates":{},
        "unit_locations": {"method": "monopolar_triangulation"},
        "spike_amplitudes": {},
        "spike_locations":{"method":"monopolar_triangulation"},
        "template_metrics":{"include_multi_channel_metrics":True},
        "template_similarity":{},
        "quality_metrics":{}
         }
    )
    # sorting_analyzer.compute("waveforms")
    # compute_dict = {
    #     "principal_components": {"n_components": 3},
    #     "templates": {"operators": ["average"]},#needs to check if average or ms_before or ms after is better according to Guided Spikeinterface
    # }
    # drift_ptps, drift_stds, drift_mads = sqm.compute_drift_metrics(sorting_analyzer=sorting_analyzer,peak_sign="neg")
    #
    # sorting_analyzer.compute(
    #     [
    #         #"amplitude_scalings",
    #     ]


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
    phy_folder_name = "phy" + sorter_suffix
    report_folder_name = "report" + sorter_suffix
    remove_excess_spikes = True
    postprocess_with_unwhitening_recording=analysis_methods.get("postprocess_with_unwhitening_recording",False)
    load_previous_methods=analysis_methods.get("load_previous_methods",False)
    if load_previous_methods:
        previous_methods_file=find_file(oe_folder / sorting_folder_name, "analysis_methods_dictionary_backup.json")
        if previous_methods_file!=None:
            with open(previous_methods_file, "r") as f:
                print(f"load analysis methods from previous file {previous_methods_file}")
                previous_analysis_methods = json.loads(f.read())
            analysis_methods.update(previous_analysis_methods)
        else:
            print("previous analysis methods file is not found. Use the current one.")

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
        # sorting_duduplicated = scur.remove_duplicated_spikes(sorting_spikes)
        # sorting_spikes = sorting_duduplicated
        print(scur.find_redundant_units(sorting_spikes))
        unit_labels = sorting_spikes.get_property("quality")
        recording_saved = get_preprocessed_recording(oe_folder,analysis_methods)
        analysis_methods.update({"load_existing_motion_info": True})
        recording_saved=spre.astype(recording_saved,np.float32)
        recording_corrected_dict=motion_correction_shankbyshank(recording_saved,oe_folder,analysis_methods)
        if len(recording_corrected_dict)>1:
            recording_for_analysis=si.aggregate_channels(recording_corrected_dict)
        else:
            recording_for_analysis=recording_corrected_dict[0]

        sorting_analyzer = si.create_sorting_analyzer(
            sorting=sorting_spikes,
            recording=recording_for_analysis,
            sparse=False,  # default
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
            recording_for_analysis,
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
        recording_saved = get_preprocessed_recording(oe_folder,analysis_methods)
        analysis_methods.update({"load_existing_motion_info": True})
        recording_saved=spre.astype(recording_saved,np.float32)
        recording_corrected_dict=motion_correction_shankbyshank(recording_saved,oe_folder,analysis_methods)
        if len(recording_corrected_dict)>1 and postprocess_with_unwhitening_recording==True:
            recording_for_analysis=si.aggregate_channels(recording_corrected_dict)
        elif len(recording_corrected_dict)==1:
            recording_for_analysis=recording_corrected_dict[0]
        elif postprocess_with_unwhitening_recording==False and len(recording_corrected_dict)>1:
            recording_corrected_dict = spre.whiten(recording=recording_corrected_dict,mode="local",radius_um=150)
            recording_for_analysis=si.aggregate_channels(recording_corrected_dict)
        else:
            recording_for_analysis = spre.whiten(recording=recording_corrected_dict[0],mode="local",radius_um=150)
        if remove_excess_spikes:
            sorting_spikes=scur.remove_duplicated_spikes(sorting_spikes,censored_period_ms=0.3,method="keep_first_iterative")
            sorting_spikes = scur.remove_excess_spikes(
                sorting_spikes, recording_for_analysis
            )
            print(f"find redundant units: {scur.find_redundant_units(sorting_spikes)}") #sorting_spikes= scur.remove_redundant_units(sorting_spikes,align=False) remove redundant on spikeinterface-gui
        sorting_analyzer = si.create_sorting_analyzer(
            sorting=sorting_spikes,
            recording=recording_for_analysis,
            sparse=True,  # default
            format="memory",  # default
        )
        calculate_analyzer_extension(sorting_analyzer)
        if analysis_methods.get("save_analyser_to_disc")==True:
            sorting_analyzer.save_as(folder=oe_folder / analyser_folder_name,format="zarr")
        if analysis_methods.get("export_to_phy") == True:
            if Path(oe_folder / phy_folder_name).exists() and analysis_methods.get("overwrite_existing_phy") == False:
                _ = se.read_phy(
                oe_folder / phy_folder_name)
            else:
                sep.export_to_phy(
                    sorting_analyzer,
                    output_folder=oe_folder / phy_folder_name,
                    compute_amplitudes=True,
                    compute_pc_features=True,
                    copy_binary=False,
                    remove_if_exists=True,
                )
                print(f"postprocessing is done. Export the files to phy to do manual curation")
            print(f"get potential auto merge{scur.get_potential_auto_merge(sorting_analyzer)}")
        else:
            print("try out some automatic curation")
            quality_metrics=sqm.compute_quality_metrics(sorting_analyzer)
            isi_viol_thresh = 0.5
            amp_cutoff_thresh = 0.1
            rp_thresh=0.2
            sd_thresh=1.5
            snr_thresh=1.1
            firing_rate_thresh=1.0
            #curation_rule = f"amplitude_cutoff < {amp_cutoff_thresh} & isi_violations_ratio < {isi_viol_thresh}" #the one used in spikeinterface notebook
            curation_rule = f"firing_rate > {firing_rate_thresh} & snr > {snr_thresh} & rp_contamination < {rp_thresh} & sd_ratio < {sd_thresh}" #the one used in Guided Spikeinterface Hands-on
            good_metrics=quality_metrics.query(curation_rule)
            curated_unit_ids=list(good_metrics.index)
            print(f"potential good units: {curated_unit_ids}, they are defined by some fixed threshold")
        ### the following information should be calculated in calculate_analyzer_extension already
            # print(sqm.get_quality_metric_list())
            # sqm.compute_quality_metrics(sorting_analyzer,metric_names=["isolation_distance","d_prime","snr","sd_ratio"])   
    else:
        return print(
            f"{sorting_folder_name} is not found. Noting can be done here without some putative spikes..."
        )

    noise_neuron_labels = scur.auto_label_units(sorting_analyzer = sorting_analyzer,repo_id ="SpikeInterface/UnitRefine_noise_neural_classifier",trust_model=True) #or ['numpy.dtype']
    # Apply the noise/not-noise model
    noise_units = noise_neuron_labels[noise_neuron_labels['prediction']=='noise']
    #noise_units.to_csv(oe_folder / sorting_folder_name/'predicted_noise_units.csv')
    print(noise_units)
    analyzer_neural = sorting_analyzer.remove_units(noise_units.index)
    # Apply the sua/mua model
    sua_mua_labels = scur.auto_label_units(
        sorting_analyzer=analyzer_neural,
        repo_id="SpikeInterface/UnitRefine_sua_mua_classifier",
        trust_model=True,
    )
    all_labels = pd.concat([sua_mua_labels, noise_units]).sort_index()
    all_labels.to_csv(oe_folder / sorting_folder_name/'predicted_sua_mua.csv')

    if (analysis_methods.get("load_curated_spikes") == True
        and (oe_folder / phy_folder_name).is_dir()):
        #evalute the performance between automated labels and manual labels
        model,model_info=scur.load_model(repo_id="SpikeInterface/UnitRefine_noise_neural_classifier",trusted=['numpy.dtype'])
        human_labels = sorting_analyzer.sorting.get_property('quality')
        label_conversion = model_info['label_conversion']
        predictions = noise_neuron_labels['prediction']
        conf_matrix = confusion_matrix(human_labels, predictions)
        # Calculate balanced accuracy for the confusion matrix
        balanced_accuracy = balanced_accuracy_score(human_labels, predictions)

        plt.imshow(conf_matrix)
        for (index, value) in np.ndenumerate(conf_matrix):
            plt.annotate( str(value), xy=index, color="white", fontsize="15")
        plt.xlabel('Predicted Label')
        plt.ylabel('Human Label')
        plt.xticks(ticks = [0, 1], labels = list(label_conversion.values()))
        plt.yticks(ticks = [0, 1], labels = list(label_conversion.values()))
        plt.title('Predicted vs Human Label')
        plt.suptitle(f"Balanced Accuracy: {balanced_accuracy}")
        plt.show()

        confidences = noise_neuron_labels['probability']

        # Make dataframe of human label, model label, and confidence
        label_df = pd.DataFrame(data = {
            'human_label': human_labels,
            'decoder_label': predictions,
            'confidence': confidences},
            index = sorting_analyzer.sorting.get_unit_ids())

        # Calculate the proportion of agreed labels by confidence decile
        label_df['model_x_human_agreement'] = label_df['human_label'] == label_df['decoder_label']

        p_agreement_sorted, p_agreement_moving_avg = calculate_moving_avg(label_df, 'confidence', 3)

        # Plot the moving average of agreement
        plt.figure(figsize=(6, 6))
        plt.plot(p_agreement_sorted, p_agreement_moving_avg, label = 'Moving Average')
        plt.axhline(y=1/len(np.unique(predictions)), color='black', linestyle='--', label='Chance')
        plt.xlabel('Confidence'); #plt.xlim(0.5, 1)
        plt.ylabel('Proportion Agreement with Human Label'); plt.ylim(0, 1)
        plt.title('Agreement vs Confidence (Moving Average)')
        plt.legend(); plt.grid(True); plt.show()

 
    sw.plot_quality_metrics(sorting_analyzer, include_metrics=["isi_violations_ratio","snr","rp_contamination","firing_rate","sd_ratio"])
    ax = sw.plot_unit_templates(sorting_analyzer, backend="matplotlib")
    fig_name = f"preview_unit_template.png"
    fig_dir = oe_folder / fig_name
    ax.figure.savefig(fig_dir)

    if analysis_methods.get("export_report") == True:
        sep.export_report(
            sorting_analyzer, output_folder=oe_folder / report_folder_name
        )
    return sorting_analyzer


if __name__ == "__main__":
    #thisDir = r"Y:\GN25019\250524\2025-05-24_15-11-49"
    thisDir = r"Y:\GN25060\251130\coherence\session1\2025-11-30_14-25-01"
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    _=si2phy(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
