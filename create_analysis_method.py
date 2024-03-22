import json

file_name = "analysis_methods_dictionary.json"
analysis_methods = {
    "experimenter":"chiyu",
    "sorter_name":"kilosort3",
    "analyse_good_channels_only":True,
    "load_raw_traces":False,
    "analyse_entire_recording": True,
    "save_prepocessed_file":False,
    "load_prepocessed_file":True,
    "save_sorting_file":False,
    "load_sorting_file":False,
    "overwrite_curated_dataset": True,
    "graph_colour_code": ["r", "y", "m", "c", "k", "b", "g", "r"],
    "debug_mode": False,
    "isi_onset_log": False,
    "filtering_method": "sg_filter",
    "plotting_tbt_overview": True,
    "load_experiment_condition_from_database": True,
    "select_animals_by_condition": True,
    "analysis_by_stimulus_type": False,
    "trackball_radius": 40,
    "frame_rate": 144,
    "stim_duration": 20,
    "interval_duration": 10,
    "stim_type": [
        -100,
        -75,
        -50,
        -25,
        0,
        25,
        50,
        75,
        100,
    ],
}  # plue value representing clockwise, counterclockwise is minus, then the rest is coherence leve
json_string = json.dumps(analysis_methods, indent=1)
with open(file_name, "w") as f:
    f.write(json_string)
