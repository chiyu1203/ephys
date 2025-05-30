# Import utilities for loading and plotting data
# Import spectral variance functions
import time,json
from numpy import array
from pathlib import Path
from open_ephys.analysis import Session
from neurodsp.spectral import compute_spectral_hist, compute_scv, compute_scv_rs
# Import function to compute power spectra
from neurodsp.spectral import compute_spectrum, rotate_powerlaw
from neurodsp.plts.time_series import plot_time_series
from neurodsp.plts.spectral import (plot_spectral_hist, plot_scv,
                                    plot_scv_rs_lines, plot_scv_rs_matrix,plot_power_spectra)
from neurodsp.plts.utils import check_ax, save_figure
from neurodsp.utils import create_times
def calculate_psd(sig,fs,times,num,fig_dir):
        #plot_time_series(times, sig)
        # Mean of spectrogram (Welch)
    freq_mean, psd_mean = compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)
    # Median of spectrogram ("median Welch")
    freq_med, psd_med = compute_spectrum(sig, fs, method='welch', avg_type='median', nperseg=fs*2)
    # Median filtered spectrum
    freq_mf, psd_mf = compute_spectrum(sig, fs, method='medfilt')
    plot_power_spectra([freq_mean[:200], freq_med[:200], freq_mf[100:10000]],
                [psd_mean[:200], psd_med[:200], psd_mf[100:10000]],
                ['Welch', 'Median Welch', 'Median Filter FFT'])
    

def run_noise_check(thisDir,analysis_methods):
    clean_up_line_noise=False
    oe_folder = Path(thisDir)
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    session = Session(oe_folder)
    recording = session.recordnodes[0].recordings[0]
    fs = recording.info["continuous"][0]["sample_rate"]

    if analysis_methods.get("analyse_entire_recording",True) == True:
        data_of_interest = len(recording.continuous[0].sample_numbers)
    else:
        data_of_interest = int(fs) * 60 # default to get 1 minute of recording
    data = recording.continuous[0].get_samples(
        start_sample_index=0, end_sample_index=data_of_interest
    )
    
    times = create_times(data.shape[0] / fs, fs)
    if clean_up_line_noise:
        ## the matlab engine needs to be initialised first. By going to the matlab and typed 
        '''
        matlab.engine.shareEngine
        matlab.engine.engineName
        Then copy the engine number and paste it in the code below
        '''
        import matlab.engine
        m=matlab.engine.connect_matlab('MATLAB_9924')
    # 
    # similar procedure when using cleanline #EEG = m.pop_cleanline(data, 'Bandwidth',2,'SignalType','Channels','ComputeSpectralPower',1,'LineFrequencies',50)
    # or in the case of using notch filter mne.filter.notch_filter
    for i in range(32):
        sig = data[:, i]
        if clean_up_line_noise:
            zapline_output = m.clean_data_with_zapline_plus(sig,fs,'noisefreqs','line','plotResults',0)
            sig = array(zapline_output)[0]
            file_name=f"power_spectra_cleaned{i}.png"
        else:
            file_name=f"power_spectra_raw{i}.png"
        calculate_psd(sig, fs, times, i, oe_folder)
        save_figure(file_name,file_path=oe_folder,close=True)
if __name__ == "__main__":
    #thisDir = r"Y:\GN25017\250518\gratings\session1\2025-05-18_21-32-15"
    thisDir = r"Y:\GN25021\250529\2025-05-29_18-47-15"
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    run_noise_check(thisDir,json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")