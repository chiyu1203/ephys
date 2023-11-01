##this data comes from https://zenodo.org/record/21589
'''
attrs of interest:
f['Continuous_1'].attrs['log_file_content']
'Experiment Parameters:\n  number of trials: 90\n  trial length: 29 sec\n  delay to odor: 3 sec\n  odor duration: 1000 msec\n  interval between start of trials: 30 sec\n  master8 channel: 8
f['Citral_1'].attrs['log_file_content']
'Experiment Parameters:\n  number of trials: 50\n  trial length: 29 sec\n  delay to odor: 3 sec\n  odor duration: 1000 msec\n  interval between start of trials: 30 sec\n  master8 channel: 8
'''
import os, fnmatch
import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
from psd_analysis import calculate_psd
from neurodsp.utils import create_times

def main(thisDir,analysis_methods):
    fs=15000
    file_name="locust20010124b_part1.hdf5"
    thisH5=os.path.join(thisDir,file_name)
    with h5py.File(thisH5,"r") as f:
        for this_session in f.keys():
            for this_trial in list(f[this_session]):
                for this_channel in list(f[this_session][this_trial]):
                    data=np.array(f[this_session][this_trial][this_channel])
                    times = create_times(data.shape[0]/fs, fs)
                    recording_name=this_session+this_trial+this_channel
                    calculate_psd(data,fs,times,recording_name,thisDir)

if __name__ == "__main__":
    thisDir = "C:/Users/neuroLaptop/Documents/Open Ephys/Pouzat2002"
    analysis_methods = {
        "Overwrite_curated_dataset": True,
        "Reanalyse_data": True,
        "Fig_dir":"Z:/DATA/experiment_openEphys/GN00001",
        "Analye_entire_recording":True,
        "Plot_trace": False,
        "Debug_mode": True,
    }
    ##Time the function
    tic = time.perf_counter()
    main(thisDir, analysis_methods)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")