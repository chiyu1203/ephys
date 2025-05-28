import json

file_name = "analysis_methods_dictionary.json"
electrophysiology_methods = {
    "experimenter": "chiyu",
    "save_output": True,
    "overwrite_curated_dataset": True,
    "plot_traces":False,
    "probe_type":"P2",
    "motion_corrector":"dredge",
    "sorter_name": "kilosort4",
    "remove_dead_channels":True,
    "analyse_good_channels_only": True,
    "interpolate_noisy_channels":True,
    "load_raw_traces": False,
    "tmin_tmax": [0.0,-1.0],
    "skip_motion_correction":True,
    "load_existing_motion_info":False,
    "save_prepocessed_file": False,
    "load_prepocessed_file": True,
    "save_sorting_file": True,
    "load_sorting_file": False,
    "save_analyser_to_disc": True,
    "load_analyser_from_disc": False,
    "extract_waveform_sparse": False,
    "extract_waveform_sparse_explicit": False,
    "export_to_phy": True,
    "overwrite_existing_phy": True,
    "load_curated_spikes": True,
    "export_report": False,
    "include_MUA": True,
}  # plue value representing clockwise, counterclockwise is minus, then the rest is coherence leve
json_file = f"./{file_name}"
if isinstance(json_file, dict):
    analysis_methods = json_file
else:
    with open(json_file, "r") as f:
        print(f"load analysis methods from file {json_file}")
        analysis_methods = json.loads(f.read())
analysis_methods.update(electrophysiology_methods)
json_string = json.dumps(analysis_methods, indent=1)
with open(file_name, "w") as f:
    print(f"update electrophysiology methods to file {json_file}")
    f.write(json_string)