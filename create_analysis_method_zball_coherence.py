import json

file_name = "analysis_methods_dictionary.json"
behaviour_methods = {
    "exp_place": "Zball",
    "experiment_name": "coherence",
    "temperature_data_options": "trial",
    "stim_variable2":'Duration',
    "overwrite_curated_dataset": True,
    "graph_colour_code": ["r", "y", "m", "c", "k", "b", "g", "r"],
    "save_output": True,
    "fictrac_posthoc_analysis": True,
    "use_led_to_align_stimulus_timing": True,
    "align_with_isi_onset": False,
    "mark_jump_as_nan": False,
    "active_trials_only": True,
    "filtering_method": "sg_filter",
    "plotting_tbt_overview": True,
    "plotting_trajectory": True,
    "plotting_event_related_trajectory": True,
    "plotting_deceleration_accerleration": False,
    "plotting_position_dependant_fixation": False,
    "plotting_optomotor_response": False,
    "load_experiment_condition_from_database": True,
    "select_animals_by_condition": False,
    "analysis_by_stimulus_type": True,
    "stationary_phase_before_motion": False,
    "duration_for_optomotor_index": 5,
    "yaw_axis":"z",
    "camera_fps": 100,
    "trackball_radius": 50,
    "monitor_fps": 144,
    "prestim_duration": 190,
    "stim_duration": [5,25,50],
    "interval_duration": [5,25,50],
    "analysis_window": [
        -5,
        50
    ],
    "stim_type": [
        -100,
        -50,
        -20,
        0,
        20,
        50,
        100,
    ],
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
