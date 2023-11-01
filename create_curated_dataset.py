from open_ephys.analysis import Session
from psd_analysis import calculate_psd
from neurodsp.utils import create_times
import os, fnmatch
import time
import numpy as np
import matplotlib.pyplot as plt


# Define frequency range across which to model the spectrum
# Model the power spectrum with FOOOF, and print out a report
# fm.report(freqs, spectrum, freq_range)
def find_file(thisDir, pattern):
    file_check = fnmatch.filter(os.listdir(thisDir), pattern)
    if len(file_check) == 0:
        print(f"no {pattern} found in {thisDir}. Let's leave this programme")
        return None
    elif len(file_check) == 1:
        return os.path.join(thisDir, file_check[0])
    else:
        vid_list = []
        for i in range(len(file_check)):
            vid_list.append(os.path.join(thisDir, file_check[i]))
        return vid_list


def detectTTL(event_dir, analysis_methods):
    this_file = np.load(event_dir)
    print(this_file)


def main(thisDir, analysis_methods):
    pd_pattern = "PD*.csv"
    this_PD = find_file(thisDir, pd_pattern)

    trial_pattern = "trial*.csv"
    this_trial_info = find_file(thisDir, trial_pattern)

    video_pattern = "video*.avi"
    this_video = find_file(thisDir, video_pattern)
    path_to_event = r"events\OE_FPGA_Acquisition_Board-100.Rhythm Data\TTL"
    event_dir = os.path.join(thisDir, path_to_event)
    event_pattern = "timestamps.npy"
    this_event = find_file(event_dir, event_pattern)
    detectTTL(this_event, analysis_methods)
    session = Session(thisDir)
    print(session)
    print(session.recordnodes)
    # recordnode = session.recordnodes[0]
    recording = session.recordnodes[0].recordings[0]
    recording.add_sync_line(
        1,  # TTL line number
        100,  # processor ID
        "Probe-A-LFP",  # stream name
        main=True,
    )  # align to the main stream
    recording.compute_global_timestamps()
    fs = recording.info["continuous"][0]["sample_rate"]
    if analysis_methods.get("Analye_entire_recording") == True:
        data_of_interest = len(recording.continuous[0].sample_numbers)
    else:
        data_of_interest = int(fs) * 3
    data = recording.continuous[0].get_samples(
        start_sample_index=0, end_sample_index=data_of_interest
    )
    times = create_times(data.shape[0] / fs, fs)
    fig_out = analysis_methods.get("Fig_dir")
    for i in range(32):
        sig = data[:, i]
        calculate_psd(sig, fs, times, i, fig_out)

    # print(data)
    # plt.close('all')
    # fig = plt.figure(figsize=(6,6))
    # ax = fig.add_subplot(111)
    # ax.axis('equal')
    # ax.plot(data[:,0], 'ro', label='test data', zorder=1)
    # svg_plot = "tracking_trace.svg"
    # thisPlot_svg = os.path.join(thisDir, svg_plot)
    # jpg_plot = "tracking_trace.jpg"
    # thisPlot_jpg = os.path.join(thisDir, jpg_plot)

    # if analysis_methods.get("Debug_mode") == True:
    #     print("no picture is saved in debug mode")
    # else:
    #     fig.savefig(thisPlot_svg)
    #     fig.savefig(thisPlot_jpg)
    # #print(session.recordnodes[0].recordings[0])
    # return data,fig


if __name__ == "__main__":
    # thisDir = "C:/Users/neuroLaptop/Documents/Open Ephys/2023-06-06_21-51-30/"
    thisDir = r"Z:\Users\chiyu\sync_test"
    analysis_methods = {
        "Overwrite_curated_dataset": True,
        "Reanalyse_data": True,
        "Fig_dir": "Z:/DATA/experiment_openEphys/GN00001",
        "Analye_entire_recording": True,
        "Plot_trace": False,
        "Debug_mode": True,
    }
    ##Time the function
    tic = time.perf_counter()
    main(thisDir, analysis_methods)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
