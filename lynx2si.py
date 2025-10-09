import time, os, json, warnings
import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.sorters as ss
import spikeinterface.qualitymetrics as sq
import spikeinterface.exporters as sep
#from spikeinterface.sortingcomponents.motion_interpolation import interpolate_motion
#from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks

import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
##from brainbox.population.decode import get_spike_counts_in_bins
##from brainbox.task.trials import get_event_aligned_raster, get_psth
##from brainbox.ephys_plots import scatter_raster_plot
import numpy as np
from pathlib import Path
import probeinterface as pi
from probeinterface.plotting import plot_probe
import numcodecs
warnings.simplefilter("ignore")
from raw2si import generate_sorter_suffix
import spikeinterface.curation as scur
def raw2si(thisDir, json_file):
    oe_folder=Path(thisDir)
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    this_sorter=analysis_methods.get("sorter_name")
    this_experimenter=analysis_methods.get("experimenter")
    sorter_suffix=generate_sorter_suffix(this_sorter)
    result_folder_name="results"+sorter_suffix
    sorting_folder_name="sorting"+sorter_suffix
    n_cpus = os.cpu_count()
    n_jobs = n_cpus - 4
    stream_IDs=['2', '0', '3', '1']
    job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)

    full_raw_rec = se.read_neuralynx(oe_folder,stream_id=stream_IDs[0])
    fs = full_raw_rec.get_sampling_frequency()
    if analysis_methods.get("load_raw_traces")==True:
        trace_snippet = full_raw_rec.get_traces(start_frame=int(fs*0), end_frame=int(fs*2))
    
    manufacturer = 'cambridgeneurotech'
    probe_name = 'ASSY-37-P-2'

    probe = pi.get_probe(manufacturer, probe_name)
    print(probe)
    probe.wiring_to_device('ASSY-116>RHD2132')
    probe.to_dataframe(complete=True).loc[:, ["contact_ids", "shank_ids", "device_channel_indices"]]
            #drop AUX channels here
    raw_rec = full_raw_rec.set_probe(probe)



if __name__ == "__main__":
    thisDir = r"D:\neuralynx\2025-01-20_15-43-10"
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    raw2si(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")