# Import utilities for loading and plotting data
# Import spectral variance functions
from neurodsp.spectral import compute_spectral_hist, compute_scv, compute_scv_rs
# Import function to compute power spectra
from neurodsp.spectral import compute_spectrum, rotate_powerlaw
from neurodsp.plts.time_series import plot_time_series
from neurodsp.plts.spectral import (plot_spectral_hist, plot_scv,
                                    plot_scv_rs_lines, plot_scv_rs_matrix,plot_power_spectra)
from neurodsp.plts.utils import check_ax, save_figure
from fooof import FOOOF

# Initialize FOOOF object
def fooof_analysis():
    fm = FOOOF()
    freq_range = [3, 40]
def calculate_psd(sig,fs,times,num,fig_dir):
        #plot_time_series(times, sig)
        # Mean of spectrogram (Welch)
        freq_mean, psd_mean = compute_spectrum(sig, fs, method='welch', avg_type='mean', nperseg=fs*2)

        # Median of spectrogram ("median Welch")
        freq_med, psd_med = compute_spectrum(sig, fs, method='welch', avg_type='median', nperseg=fs*2)

        # Median filtered spectrum
        freq_mf, psd_mf = compute_spectrum(sig, fs, method='medfilt')
        
        file_name=f"power_spectra_{num}.png"
        plot_power_spectra([freq_mean[:200], freq_med[:200], freq_mf[100:10000]],
                   [psd_mean[:200], psd_med[:200], psd_mf[100:10000]],
                   ['Welch', 'Median Welch', 'Median Filter FFT'])

        save_figure(file_name,file_path=fig_dir,close=True)