import json

file_name = "analysis_methods_dictionary.json"
behaviour_methods = {
    "exp_place": "Zball",
    "experiment_name": "choices",
    "temperature_data_options": "trial",
    "load_previous_methods": False,
    "overwrite_curated_dataset": True,
    "graph_colour_code": ["r", "y", "m", "c", "k", "b", "g", "r"],
    "save_output": True,
    "fictrac_posthoc_analysis": False,
    "use_led_to_align_stimulus_timing": False,
    "align_with_isi_onset": False,
    "mark_jump_as_nan": False,
    "active_trials_only": True,
    "filtering_method": "rolling_median",
    "load_experiment_condition_from_database": True,
    "select_animals_by_condition": False,
    "analysis_by_stimulus_type": True,
    "stationary_phase_before_motion": False,
    "yaw_axis":"z",
    "camera_fps": 144,
    "trackball_radius": 5,
    "monitor_fps": 144,
    "prestim_duration": 60,
    "stim_duration": [30],
    "interval_duration": [20,30,40],
    "analysis_window": [
        -2,
        30
    ],
    "event_of_interest": [
        "stim_onset",
        "turn_ccw_onset",
        "turn_cw_onset",
        "turn_onset",
    ],
    "stim_variables": [
        "LocustTexture1",
        "LocustTexture2",
        "A1",
        "A2",
        "R1",
        "R2",
    ],
    "zeta_variables1": [],
    "zeta_variables2": [],
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
