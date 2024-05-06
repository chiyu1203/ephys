# Modified the code from the following source so that functions can be called from the ephy pipeline as a module
# Chi-Yu Lee 07/05/2024


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

# from tkinter.filedialog import askdirectory, askopenfilename

# Inputs from user
################################################################################
############################ USER INPUT SECTION ################################
################################################################################

# NP events come in as indexed values, with 1 to indicate when a TTL pulse changed
# on to off (directionality). Other DAQ files (like LJ) come in 'raw' digital format,
# with 0 to indicate TTL off state, and 1 to indicate on state.
raw_data_format = False  # Set raw_data_format to True for LJ-like data, False for NP.
signals_column = 0  # Column in which sorted timestamps or raw barcode data
# appears (Base zero; 1st column = 0)
expected_sample_rate = 30000  # In Hz. Generally set to 2k Hz for the LabJack or
# 30k Hz for the Neuropixel. Choose based on your DAQ's sample rate.
global_tolerance = 0.20  # The fraction (percentage) of tolerance allowed for
# duration measurements.
# (Ex: If global_tolerance = 0.2 and ind_wrap_duration = 10, then any signal
# change between 8-12 ms long will be considered a barcode wrapper.)

# Output Variables
barcodes_name = "barcode_oe"  # Name of your output file
save_npy = True  # Save the barcodes data in .npy format (needed for alignment)
save_csv = False  # Save barcodes data in .csv format

# General variables; make sure these align with the timing format of
# your Arduino-generated barcodes.
nbits = 32
inter_barcode_interval = 5000  # In milliseconds
ind_wrap_duration = 10  # In milliseconds
ind_bar_duration = 30  # In milliseconds

################################################################################
############################ END OF USER INPUT #################################
################################################################################

###########################################################
### Set Global Variables/Tolerances Based on User Input ###
###########################################################

wrap_duration = 3 * ind_wrap_duration  # Off-On-Off
total_barcode_duration = nbits * ind_bar_duration + 2 * wrap_duration

# Tolerance conversions
min_wrap_duration = ind_wrap_duration - ind_wrap_duration * global_tolerance
max_wrap_duration = ind_wrap_duration + ind_wrap_duration * global_tolerance
min_bar_duration = ind_bar_duration - ind_bar_duration * global_tolerance
max_bar_duration = ind_bar_duration + ind_bar_duration * global_tolerance
sample_conversion = 1000 / expected_sample_rate  # Convert sampling rate to msec


##########################################################
### Select Data Input File / Barcodes Output Directory ###
##########################################################
def extract_barcodes(barcodes_dir, signals_numpy_data):
    # try:
    #     signals_file = Path(
    #         askopenfilename(
    #             title="Select Your Input Data File",
    #         )
    #     )  # shows dialog box and return the path
    # except:
    #     print("No File Chosen")
    #     sys.exit()

    # if save_npy or save_csv:
    #     try:
    #         barcodes_dir = Path(
    #             askdirectory(
    #                 title="Select Barcodes Output Folder",
    #             )
    #         )  # Output dir
    #     except:
    #         print("No Output Directory Chosen")
    #         sys.exit()
    # else:
    #     print(
    #         "No outputs selected, extracted barcodes will not be saved. What are you trying to do?"
    #     )

    ##############################################
    ### Signals Data Extraction & Manipulation ###
    ##############################################
    try:
        # signals_numpy_data = np.load(signals_file)
        signals_located = True
    except:
        signals_numpy_data = ""
        print("Signals .npy file not located; please check your filepath")
        signals_located = False

    # Check whether signals_numpy_data exists; if not, end script with sys.exit().
    if signals_located:
        # LJ = If data is in raw format and has not been sorted by "peaks"
        if raw_data_format:

            # Extract the signals_column from the raw data
            barcode_column = signals_numpy_data[:, signals_column]
            barcode_array = barcode_column.transpose()
            # Extract the indices of all events when TTL pulse changed value.
            event_index, _ = find_peaks(np.abs(np.diff(barcode_array)), height=0.9)
            # Convert the event_index to indexed_times to align with later code.
            indexed_times = event_index  # Just take the index values of the raw data

        # NP = Collect the pre-extracted indices from the signals_column.
        else:
            indexed_times = signals_numpy_data

        # Find time difference between index values (ms), and extract barcode wrappers.
        events_time_diff = np.diff(indexed_times) * sample_conversion  # convert to ms
        wrapper_array = indexed_times[
            np.where(
                np.logical_and(
                    min_wrap_duration < events_time_diff,
                    events_time_diff < max_wrap_duration,
                )
            )[0]
        ]

        # Isolate the wrapper_array to wrappers with ON values, to avoid any
        # "OFF wrappers" created by first binary value.
        false_wrapper_check = (
            np.diff(wrapper_array) * sample_conversion
        )  # Convert to ms
        # Locate indices where two wrappers are next to each other.
        false_wrappers = np.where(false_wrapper_check < max_wrap_duration)[0]
        # Delete the "second" wrapper (it's an OFF wrapper going into an ON bar)
        wrapper_array = np.delete(wrapper_array, false_wrappers + 1)

        # Find the barcode "start" wrappers, set these to wrapper_start_times, then
        # save the "real" barcode start times to signals_barcode_start_times, which
        # will be combined with barcode values for the output .npy file.
        wrapper_time_diff = np.diff(wrapper_array) * sample_conversion  # convert to ms
        barcode_index = np.where(wrapper_time_diff < total_barcode_duration)[0]
        wrapper_start_times = wrapper_array[barcode_index]
        signals_barcode_start_times = (
            wrapper_start_times - ind_wrap_duration / sample_conversion
        )
        # Actual barcode start is 10 ms before first 10 ms ON value.

        # Using the wrapper_start_times, collect the rest of the indexed_times events
        # into on_times and off_times for barcode value extraction.
        on_times = []
        off_times = []
        for idx, ts in enumerate(indexed_times):  # Go through indexed_times
            # Find where ts = first wrapper start time
            if ts == wrapper_start_times[0]:
                # All on_times include current ts and every second value after ts.
                on_times = indexed_times[idx::2]
                off_times = indexed_times[idx + 1 :: 2]  # Everything else is off_times

        # Convert wrapper_start_times, on_times, and off_times to ms
        wrapper_start_times = wrapper_start_times * sample_conversion
        on_times = on_times * sample_conversion
        off_times = off_times * sample_conversion

        signals_barcodes = []
        for start_time in wrapper_start_times:
            oncode = on_times[
                np.where(
                    np.logical_and(
                        on_times > start_time,
                        on_times < start_time + total_barcode_duration,
                    )
                )[0]
            ]
            offcode = off_times[
                np.where(
                    np.logical_and(
                        off_times > start_time,
                        off_times < start_time + total_barcode_duration,
                    )
                )[0]
            ]
            curr_time = (
                offcode[0] + ind_wrap_duration
            )  # Jumps ahead to start of barcode
            bits = np.zeros((nbits,))
            interbit_ON = False  # Changes to "True" during multiple ON bars

            for bit in range(0, nbits):
                next_on = np.where(
                    oncode >= (curr_time - ind_bar_duration * global_tolerance)
                )[0]
                next_off = np.where(
                    offcode >= (curr_time - ind_bar_duration * global_tolerance)
                )[0]

                if next_on.size > 1:  # Don't include the ending wrapper
                    next_on = oncode[next_on[0]]
                else:
                    next_on = start_time + inter_barcode_interval

                if next_off.size > 1:  # Don't include the ending wrapper
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

            for bit in range(0, nbits):  # least sig left
                barcode += bits[bit] * pow(2, bit)

            signals_barcodes.append(barcode)

    else:  # If signals_located = False
        sys.exit("Data not found. Program has stopped.")

    ################################################################
    ### Print out final output and save to chosen file format(s) ###
    ################################################################

    # Create merged array with timestamps stacked above their barcode values
    signals_time_and_bars_array = np.vstack(
        (signals_barcode_start_times, np.array(signals_barcodes))
    )
    print("Final Ouput: ", signals_time_and_bars_array)

    time_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if save_npy:
        output_file = barcodes_dir / (barcodes_name + time_now)
        np.save(output_file, signals_time_and_bars_array)

    if save_csv:
        output_file = barcodes_dir / (barcodes_name + time_now + ".csv")
        np.savetxt(output_file, signals_time_and_bars_array, delimiter=",", fmt="%s")
    return output_file, signals_time_and_bars_array
