import json

file_name = "analysis_methods_dictionary.json"
behaviour_methods = {
    "exp_place": "Zball",
    "temperature_data_options": "trial",
    "overwrite_curated_dataset": True,
    "graph_colour_code": ["r", "y", "m", "c", "k", "b", "g", "r"],
    "save_output": True,
    "fictrac_posthoc_analysis": True,
    "mark_jump_as_nan": True,
    "filtering_method": "rolling_median",
    "select_animals_by_condition": False,
    "yaw_axis":"z",
    "camera_fps": 100,
    "trackball_radius": 5,
    "analysis_window": [
        -2,
        2
    ],
    "event_of_interest": ["stop_onset","walk_straight_onset","turn_ccw_onset","turn_cw_onset","walk_onset","turn_onset"],
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
