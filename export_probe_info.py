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

    manufacturer = "cambridgeneurotech"
    if probe_type == "H10_stacked":
        probe_name="H10_stacked_probes"
        pg = pi.read_probeinterface(f"{probe_name}.json")
    else:
        if probe_type == "P2":
            probe_name = "ASSY-37-P-2"
            connector_id="ASSY-116>RHD2132"

        elif probe_type == "H5":
            probe_name = 'ASSY-77-H5'
            connector_id='ASSY-77>Adpt.A64-Om32_2x-sm-NN>RHD2164'
        probe = pi.get_probe(manufacturer, probe_name)
        probe.wiring_to_device(connector_id)
        probe.to_dataframe(complete=True).loc[
            :, ["contact_ids", "shank_ids", "device_channel_indices"]
        ]
        print(probe)
        pg = ProbeGroup()
        pg.add_probe(probe)
        #probe = stacked_probes.probes[0]

    # Multiple probes can be added to a ProbeGroup. We only have one, but a
    # ProbeGroup wrapper is still necessary for `write_prb` to work.

    # CHANGE THIS PATH to wherever you want to save your probe file.
    write_prb(f'{probe_name}.prb', pg)
if __name__ == "__main__":
    json_file = "./analysis_methods_dictionary.json"
    generate_prb(json_file)