from probeinterface import ProbeGroup, write_prb
import probeinterface as pi
import json
## this code takes json file and generates prb file for kilosort to read

def generate_prb(json_file):
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    probe_type = analysis_methods.get("probe_type")


    if probe_type == "P2":
        manufacturer = "cambridgeneurotech"
        probe_name = "ASSY-37-P-2"
        probe = pi.get_probe(manufacturer, probe_name)
        print(probe)
        probe.wiring_to_device("ASSY-116>RHD2132")
        probe.to_dataframe(complete=True).loc[
            :, ["contact_ids", "shank_ids", "device_channel_indices"]
        ]
        pg = ProbeGroup()
        pg.add_probe(probe)
    elif probe_type == "H10_stacked":
        probe_name="H10_stacked_probes"
        pg = pi.read_probeinterface(f"{probe_name}.json")
        #probe = stacked_probes.probes[0]

    # Multiple probes can be added to a ProbeGroup. We only have one, but a
    # ProbeGroup wrapper is still necessary for `write_prb` to work.

    # CHANGE THIS PATH to wherever you want to save your probe file.
    write_prb(f'{probe_name}.prb', pg)
if __name__ == "__main__":
    json_file = "./analysis_methods_dictionary.json"
    generate_prb(json_file)