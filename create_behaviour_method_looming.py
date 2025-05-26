import json

file_name = "analysis_methods_dictionary.json"
behaviour_methods = {
    "experimenter": "chiyu",
    "save_output": True,
    "overwrite_curated_dataset": True,
    "plot_traces":False,
    "exp_place": "Zball",
    "experiment_name": "looming",
    "temperature_data_options": "bonsai",
    "graph_colour_code": ["r", "y", "m", "c", "k", "b", "g", "r"],
    "fictrac_posthoc_analysis": True,
    "use_led_to_align_stimulus_timing": True,
    "align_with_isi_onset": False,
    "mark_jump_as_nan": True,
    "active_trials_only": True,
    "filtering_method": "sg_filter",
    "plotting_tbt_overview": True,
    "plotting_trajectory": False,
    "plotting_event_related_trajectory": False,
    "plotting_deceleration_accerleration": False,
    "plotting_position_dependant_fixation": False,
    "plotting_optomotor_response": True,
    "load_experiment_condition_from_database": True,
    "select_animals_by_condition": False,
    "analysis_by_stimulus_type": True,
    "camera_fps": 144,
    "trackball_radius": 50,
    "monitor_fps": 144,
    "stim_duration": [5,15,30],
    "interval_duration": 45,
    "analysis_window": [-5,10],
    "prestim_duration": 60,
}  # plue value representing clockwise, counterclockwise is minus, then the rest is coherence leve
json_file = f"./{file_name}"
if isinstance(json_file, dict):
    analysis_methods = json_file
else:
    with open(json_file, "r") as f:
        print(f"load analysis methods from file {json_file}")
        analysis_methods = json.loads(f.read())
analysis_methods.update(behaviour_methods)
json_string = json.dumps(analysis_methods, indent=1)
with open(file_name, "w") as f:
    print(f"update behaviour methods to file {json_file}")
    f.write(json_string)
