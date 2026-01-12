#adapted from https://optogeneticsandneuralengineeringcore.gitlab.io/ONECoreSite/projects/DAQSyncro/DAQSyncronization/
"""
  Optogenetics and Neural Engineering Core ONE Core
  University of Colorado, School of Medicine
  18.Nov.2021
  See bit.ly/onecore for more information, including a more detailed write up.
  extraction_barcodes.py
################################################################################
  This code takes a Numpy (.npy) data file from a DAQ system that used the
  "arduino-barcodes(-trigger).ino" Arduino script while recording data,
  extracts the index values when barcodes were initiated, calculates the
  value of these barcodes, and outputs a Numpy file that contains a 2D array of
  each index and its corresponding barcode value. This code should be run for
  each DAQ's Numpy file that were attached to the Arduino barcode generator
  during the same session so that "alignment_barcodes.py" can be run on them.
################################################################################
  USER INPUTS EXPLAINED:
    = raw_data_format = (bool) Set to "True" if the data being inputted has not
                        been filtered to just event timestamps (like in LJ data);
                        set to "False" if otherwise (like NP data from OpenEphys)
    = signals_column = (int) The column where the sorted signal timestamps or the
                       raw barcode data appears (base 0, 1st column = 0).
    = expected_sample_rate = (int) The DAQ's sample rate, in Hz, when this data
                             was collected. For example, NP runs at 30000 Hz.
    = global_tolerance = (float) The fraction (in %/100) of tolerance allowed
                         for duration measurements (ex: ind_bar_duration).
    = barcodes_name = (str) The name of the outputted file(s) that will be saved
                      to your chosen directory.
    = save_npy = (bool) Set to "True" if you want the output saved as a .npy file.
    = save_csv = (bool) Set to "True" if you want the output saved as a .csv file.

    (The user inputs below are based on your Arduino barcode generator settings)
    = nbits = (int) the number of bits (bars) that are in each barcode (not
              including wrappers).
    = inter_barcode_interval = (int) The duration of time (in milliseconds)
                               between each barcode's start.
    = ind_wrap_duration = (int) The duration of time (in milliseconds) of the
                          ON wrapper portion (default = 10 ms) in the barcodes.
    = ind_bar_duration = (int) The duration of time (in milliseconds) of each
                         bar (bit) in the barcode.
################################################################################
  References

"""

import numpy as np
import sys
from scipy.signal import find_peaks
from datetime import datetime
from pathlib import Path
from tkinter.filedialog import askdirectory, askopenfilename
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pandas as pd

def detect_filtered_transitions(file_path):
    ON_THRESHOLD = 20000
    OFF_THRESHOLD = 10000
    MIN_ON_LENGTH = 5
    MIN_OFF_LENGTH = 5

    # Load CSV
    df = pd.read_csv(file_path)

    # Extract the second column as a Series
    signal = df.iloc[:, 1]

    # Step 1: Create a raw state series with hysteresis
    # None = undefined / holding previous state
    raw_state = []

    current_state = None
    for value in signal:
        if value > ON_THRESHOLD:
            current_state = "ON"
        elif value < OFF_THRESHOLD:
            current_state = "OFF"
        raw_state.append(current_state)

    df["raw_state"] = raw_state

    # Step 2: Enforce minimum state duration
    validated_state = df["raw_state"].copy()

    i = 0
    while i < len(validated_state):
        state = validated_state.iloc[i]
        if state is None:
            i += 1
            continue

        # Count consecutive occurrences
        j = i
        while j < len(validated_state) and validated_state.iloc[j] == state:
            j += 1

        run_length = j - i

        # Invalidate runs shorter than minimum length
        if state == "ON" and run_length < MIN_ON_LENGTH:
            validated_state.iloc[i:j] = None
        elif state == "OFF" and run_length < MIN_OFF_LENGTH:
            validated_state.iloc[i:j] = None

        i = j

    df["state"] = validated_state.ffill()

    # Step 3: Detect transitions
    on_to_off = []
    off_to_on = []

    previous_state = df["state"].iloc[0]

    for idx in range(1, len(df)):
        current_state = df["state"].iloc[idx]

        if previous_state == "ON" and current_state == "OFF":
            on_to_off.append(idx)

        if previous_state == "OFF" and current_state == "ON":
            off_to_on.append(idx)

        previous_state = current_state

    # Results
    print("ON → OFF transition indices:", on_to_off)
    print("OFF → ON transition indices:", off_to_on)
    return on_to_off, off_to_on
# Inputs from user
################################################################################
############################ USER INPUT SECTION ################################
################################################################################

# NP events come in as indexed values, with 1 to indicate when a TTL pulse changed
# on to off (directionality). Other DAQ files (like LJ) come in 'raw' digital format,
# with 0 to indicate TTL off state, and 1 to indicate on state.
raw_data_format = True  # Set raw_data_format to True for LJ-like data, False for NP.
signals_column = 1 # Column in which sorted timestamps or raw barcode data
# appears (Base zero; 1st column = 0)
secondary_sample_rate=expected_sample_rate = 144 # In Hz. Generally set to 2k Hz for the LabJack or
# 30k Hz for the Neuropixel. Choose based on your DAQ's sample rate.
global_tolerance = .20 # The fraction (percentage) of tolerance allowed for
# duration measurements.
# (Ex: If global_tolerance = 0.2 and ind_wrap_duration = 10, then any signal
# change between 8-12 ms long will be considered a barcode wrapper.)

# Output Variables
#barcodes_name = 'oe_barcode' # Name of your output file
barcodes_name = 'video_barcode' # Name of your output file
save_npy = True  # Save the barcodes data in .npy format (needed for alignment)
save_csv = False  # Save barcodes data in .csv format

# General variables; make sure these align with the timing format of
# your Arduino-generated barcodes.
nbits = 32
inter_barcode_interval = 3000  # In milliseconds
ind_wrap_duration = 60  # the default value was 10 In milliseconds
ind_bar_duration = 60 # the default value was 30 In milliseconds

wrap_duration = 3 * ind_wrap_duration # Off-On-Off
total_barcode_duration = nbits * ind_bar_duration + 2 * wrap_duration

# Tolerance conversions
min_wrap_duration = ind_wrap_duration - ind_wrap_duration * global_tolerance
max_wrap_duration = ind_wrap_duration + ind_wrap_duration * global_tolerance
min_bar_duration = ind_bar_duration - ind_bar_duration * global_tolerance
max_bar_duration = ind_bar_duration + ind_bar_duration * global_tolerance
 # Convert sampling rate to msec

def run(signals_file):
    barcode_timestamps_row = 0 # Same for both main and secondary, because we used our own code
    barcodes_row = 1 # Same for both main and secondary
    secondary_numpy_data=np.load('video_barcodecurated.npy')
    main_numpy_data=np.load('oe_barcodecurated.npy')
    main_numpy_barcode = main_numpy_data[barcodes_row, :]
    secondary_numpy_barcode = secondary_numpy_data[barcodes_row, :]
    fig5, (ax,ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 7), tight_layout=True)
    #ax.scatter(np.arange(tmp[0,:].shape[0]),tmp[0,:])
    ax.scatter(np.arange(main_numpy_barcode.shape[0]),main_numpy_barcode)
    ax2.scatter(np.arange(secondary_numpy_barcode.shape[0]),secondary_numpy_barcode)
    fig5.savefig('barcode_pattern.png')

    main_numpy_timestamp = main_numpy_data[barcode_timestamps_row, :]
    secondary_numpy_timestamp = secondary_numpy_data[barcode_timestamps_row, :]

    # Pull the index values from barcodes shared by both groups of data
    shared_barcodes, main_index, second_index = np.intersect1d(main_numpy_barcode,
                                                secondary_numpy_barcode, return_indices=True)
    # Note: To intersect more than two arrays, use functools.reduce

    # Use main_index and second_index arrays to extract related timestamps
    main_shared_barcode_times = main_numpy_timestamp[main_index]
    secondary_shared_barcode_times = secondary_numpy_timestamp[second_index]

    # Determine slope (m) between main/secondary timestamps
    m = (main_shared_barcode_times[-1]-main_shared_barcode_times[0])/(secondary_shared_barcode_times[-1]-secondary_shared_barcode_times[0])
    # Determine offset (b) between main and secondary barcode timestamps
    b = main_shared_barcode_times[0] - secondary_shared_barcode_times[0] * m

    print('Linear conversion from secondary timestamps to main:\ny = ', m, 'x + ', b)

    # ##################################################################
    # ### Apply Linear Conversion to Secondary Data (in .npy Format) ###
    # ##################################################################

    # secondary_data_original[:,convert_timestamp_column] = secondary_data_original[:,convert_timestamp_column] * secondary_sample_rate * m + b
    # secondary_data_converted = secondary_data_original # To show conversion complete.

    if raw_data_format:
        sample_conversion = 1000 / expected_sample_rate
        on_transitions, off_transitions = detect_filtered_transitions(signals_file)
        #event_index =np.sort(np.hstack((on_transitions.index.values,off_transitions)))
        event_index =np.sort(np.hstack((off_transitions,on_transitions)))
        signals_numpy_data = genfromtxt(signals_file, delimiter=',',skip_header=1)
        # # Extract the signals_column from the raw data
        barcode_array = signals_numpy_data[:, signals_column]
        # #barcode_array = barcode_column.transpose()
        # # Extract the indices of all events when TTL pulse changed value.
        # #event_index, _ = find_peaks(np.diff(barcode_array), distance=20000)
        # event_index, _ = find_peaks(barcode_array, height=20000)
        fig5, (ax,ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 7), tight_layout=True)
        #ax.scatter(np.arange(tmp[0,:].shape[0]),tmp[0,:])
        ax.scatter(np.arange(event_index.shape[0]),event_index)
        ax2.scatter(np.arange(barcode_array.shape[0])[2000:3000],barcode_array[2000:3000])
        fig5.savefig('events_index.png')
        # Convert the event_index to indexed_times to align with later code.
        indexed_times = event_index # Just take the index values of the raw data
        events_time_diff = np.diff(indexed_times) * sample_conversion # convert to ms

    # NP = Collect the pre-extracted indices from the signals_column.
    else:
        sample_conversion = 1000
        tmp= np.load(signals_file)
        signals_numpy_data =np.sort(np.hstack((tmp[0,:],tmp[1,:])))
        indexed_times = signals_numpy_data
        events_time_diff = np.diff(indexed_times) * sample_conversion

    # Find time difference between index values (ms), and extract barcode wrappers.
    
    #events_time_diff=np.diff(tmp,axis=0)* sample_conversion

    wrapper_array = indexed_times[np.where(
                    np.logical_and(min_wrap_duration < events_time_diff,
                                events_time_diff  < max_wrap_duration))[0]]

    # Isolate the wrapper_array to wrappers with ON values, to avoid any
    # "OFF wrappers" created by first binary value.
    false_wrapper_check = np.diff(wrapper_array) * sample_conversion # Convert to ms
    # Locate indices where two wrappers are next to each other.
    false_wrappers = np.where(
                    false_wrapper_check < max_wrap_duration)[0]
    # Delete the "second" wrapper (it's an OFF wrapper going into an ON bar)
    wrapper_array = np.delete(wrapper_array, false_wrappers+1)

    # Find the barcode "start" wrappers, set these to wrapper_start_times, then
    # save the "real" barcode start times to signals_barcode_start_times, which
    # will be combined with barcode values for the output .npy file.
    wrapper_time_diff = np.diff(wrapper_array) * sample_conversion # convert to ms
    barcode_index = np.where(wrapper_time_diff < total_barcode_duration)[0]
    wrapper_start_times = wrapper_array[barcode_index]
    signals_barcode_start_times = wrapper_start_times - ind_wrap_duration / sample_conversion
    # Actual barcode start is 10 ms before first 10 ms ON value.

    # Using the wrapper_start_times, collect the rest of the indexed_times events
    # into on_times and off_times for barcode value extraction.
    on_times = []
    off_times = []
    for idx, ts in enumerate(indexed_times):    # Go through indexed_times
        # Find where ts = first wrapper start time
        if ts == wrapper_start_times[0]:
            # All on_times include current ts and every second value after ts.
            on_times = indexed_times[idx::2]
            off_times = indexed_times[idx+1::2] # Everything else is off_times

    # Convert wrapper_start_times, on_times, and off_times to ms
    wrapper_start_times = wrapper_start_times * sample_conversion
    on_times = on_times * sample_conversion
    off_times = off_times * sample_conversion

    signals_barcodes = []
    for start_time in wrapper_start_times:
        oncode = on_times[
            np.where(
                np.logical_and(on_times > start_time,
                            on_times < start_time + total_barcode_duration)
            )[0]
        ]
        offcode = off_times[
            np.where(
                np.logical_and(off_times > start_time,
                            off_times < start_time + total_barcode_duration)
            )[0]
        ]
        curr_time = offcode[0] + ind_wrap_duration # Jumps ahead to start of barcode
        bits = np.zeros((nbits,))
        interbit_ON = False # Changes to "True" during multiple ON bars

        for bit in range(0, nbits):
            next_on = np.where(oncode >= (curr_time - ind_bar_duration * global_tolerance))[0]
            next_off = np.where(offcode >= (curr_time - ind_bar_duration * global_tolerance))[0]

            if next_on.size > 1:    # Don't include the ending wrapper
                next_on = oncode[next_on[0]]
            else:
                next_on = start_time + inter_barcode_interval

            if next_off.size > 1:    # Don't include the ending wrapper
                next_off = offcode[next_off[0]]
            else:
                next_off = start_time + inter_barcode_interval

            # Recalculate min/max bar duration around curr_time
            min_bar_duration = curr_time - ind_bar_duration * global_tolerance
            max_bar_duration = curr_time + ind_bar_duration * global_tolerance

            if min_bar_duration <= next_on <= max_bar_duration:
                bits[bit] = 1
                interbit_ON = True
            elif min_bar_duration <= next_off <= max_bar_duration:
                interbit_ON = False
            elif interbit_ON == True:
                bits[bit] = 1

            curr_time += ind_bar_duration

        barcode = 0

        for bit in range(0, nbits):             # least sig left
            barcode += bits[bit] * pow(2, bit)

        signals_barcodes.append(barcode)

    ################################################################
    ### Print out final output and save to chosen file format(s) ###
    ################################################################

    # Create merged array with timestamps stacked above their barcode values
    signals_time_and_bars_array = np.vstack((signals_barcode_start_times,
                                            np.array(signals_barcodes)))
    print("Final Ouput: ", signals_time_and_bars_array)

    time_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if save_npy:
        #output_file = barcodes_name + time_now + ".npy"
        output_file = barcodes_name +"curated.npy"
        np.save(output_file, signals_time_and_bars_array)

    if save_csv:
        #output_file = barcodes_name + time_now + ".csv"
        output_file = barcodes_name +"curated.csv"
        np.savetxt(output_file, signals_time_and_bars_array,
                delimiter=',', fmt="%s")


if __name__ == "__main__":
    #signals_file = "barcode.npy"
    signals_file="meta1_2026-01-04T20_02_54.csv"

    ##Time the function
    run(signals_file)