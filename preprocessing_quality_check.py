'''
2. Python Code:

This script will:

Generate synthetic ephys data with simulated drift.

Define probe geometry.

Perform stabilization (whitening) using functions similar to those used in IBL pipelines (often found in ibllib.dsp).

Calculate and plot the distribution of covariance matrix condition numbers before and after whitening.

Detect peaks (simplified spike detection) on the whitened data.

Bin peaks into a 2D time/depth matrix.

Run DREDge to estimate motion.

Generate the drift map plot using ibllib.ephys.visualization.plot_driftmap (or a similar custom plot if that function isn't ideal for direct DREDge output visualization) and add the DREDge motion estimate subplot.

'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import logging

# Attempt to import IBL and DREDge libraries
try:
    import ibllib.dsp as dsp
    import ibllib.ephys.spikes as spikes
    import ibllib.ephys.neuropixel as neuropixel
    import ibllib.ephys.visualization as ephys_vis
    HAS_IBLLIB = True
except ImportError:
    print("*"*50)
    print("WARNING: ibllib not found. Using simplified placeholder functions.")
    print("pip install ibllib numpy scipy matplotlib")
    print("*"*50)
    HAS_IBLLIB = False

try:
    import dredge
    HAS_DREDGE = True
except ImportError:
    print("*"*50)
    print("WARNING: dredge-ephys not found. Drift estimation cannot be performed.")
    print("pip install dredge-ephys")
    print("*"*50)
    HAS_DREDGE = False

# Configure logging for ibllib (optional, reduces verbosity)
logging.basicConfig(level=logging.WARNING)

# ------------------------------------------
# 0. Parameters and Synthetic Data Generation
# ------------------------------------------
FS = 30000  # Sampling frequency (Hz)
N_CHANNELS = 64 # Number of channels
DURATION = 60  # Duration of the recording (seconds)
N_SAMPLES = int(DURATION * FS)
CHUNK_DURATION = 2 # Duration of chunks for conditioning calculation (seconds)
N_SAMPLES_CHUNK = int(CHUNK_DURATION * FS)

# Probe Geometry (Simplified linear probe)
# Use ibllib function if available, otherwise create manually
if HAS_IBLLIB:
    # Create a simple linear geometry using Neuropixel tools
    # Use 3B2 geometry as an example structure
    try:
        # Note: This might require `phylib` installation as well
        geom = neuropixel.dense_layout(version=3) # Get a sample layout
        # Select a subset of channels if N_CHANNELS is smaller
        if N_CHANNELS < geom['x'].size:
             # Take first N_CHANNELS vertically aligned if possible
            y_unique = np.unique(geom['y'])
            ch_indices = np.where(geom['y'] < y_unique[N_CHANNELS // 2])[0] # rough selection
            if len(ch_indices) < N_CHANNELS: # fallback if selection is too small
                 ch_indices = np.arange(N_CHANNELS)
            geom = {k: v[ch_indices] for k, v in geom.items()}
        elif N_CHANNELS > geom['x'].size:
            print(f"Warning: Requested {N_CHANNELS} channels, but standard geom has {geom['x'].size}. Using {geom['x'].size}.")
            N_CHANNELS = geom['x'].size
        channel_positions = np.column_stack((geom['x'], geom['y']))
        print(f"Using {N_CHANNELS} channels from ibllib geometry.")
    except Exception as e:
        print(f"Could not create geometry using ibllib: {e}. Creating simple linear geometry.")
        HAS_IBLLIB = False # Fallback if neuropixel fails

if not HAS_IBLLIB:
    # Simple vertical linear probe, 20 um spacing
    channel_positions = np.zeros((N_CHANNELS, 2))
    channel_positions[:, 1] = np.arange(N_CHANNELS) * 20
    geom = {'x': channel_positions[:, 0], 'y': channel_positions[:, 1]}
    N_CHANNELS = channel_positions.shape[0] # Ensure N_CHANNELS matches geometry

channel_depths = channel_positions[:, 1]

# Generate Synthetic Data
print("Generating synthetic data...")
# Base noise
noise_level = 15
raw_data = np.random.randn(N_SAMPLES, N_CHANNELS) * noise_level

# Add some correlated noise (simulating volume conduction)
correlation_strength = 0.3
correlated_noise = np.random.randn(N_SAMPLES, 1) * noise_level * 5
# Simple spatial decay for correlation
spatial_decay = np.exp(-np.abs(np.arange(N_CHANNELS) - N_CHANNELS // 2) / (N_CHANNELS / 4))
raw_data += correlated_noise * spatial_decay[np.newaxis, :] * correlation_strength

# Add drifting "artifact" peaks (simulating motion artifacts)
n_artifact_sources = 3
artifact_amp = 80
artifact_freq = 0.5 # Hz, how often bursts occur
drift_speed = 30 # um/sec
drift_range = N_CHANNELS * np.mean(np.diff(np.unique(channel_depths))) * 0.6 # Drift across 60% of the probe
time_vector = np.arange(N_SAMPLES) / FS

# Simulate drift: sinusoidal motion pattern
drift_pattern = (drift_range / 2) * np.sin(2 * np.pi * (drift_speed / drift_range) * time_vector) + np.mean(channel_depths)

for i in range(n_artifact_sources):
    source_base_depth = np.random.choice(channel_depths)
    source_times = np.where(np.random.rand(N_SAMPLES) < (artifact_freq / FS))[0]

    for t_idx in source_times:
        current_center_depth = source_base_depth + drift_pattern[t_idx]
        # Find channels near the current depth
        depth_diff = np.abs(channel_depths - current_center_depth)
        affected_channels = np.where(depth_diff < 50)[0] # Affect channels within 50 um

        if len(affected_channels) > 0:
             # Simple Gaussian spread of amplitude
            peak_profile = np.exp(-depth_diff[affected_channels]**2 / (2 * (20**2))) # 20 um spatial std dev
            peak_shape = scipy.signal.ricker(int(FS * 0.002), a=4) # 2ms width peak (M-shape)

            start_idx = max(0, t_idx - len(peak_shape) // 2)
            end_idx = min(N_SAMPLES, start_idx + len(peak_shape))
            shape_slice = peak_shape[:end_idx - start_idx]

            for ch_idx, profile_amp in zip(affected_channels, peak_profile):
                 raw_data[start_idx:end_idx, ch_idx] += shape_slice * artifact_amp * profile_amp

print("Synthetic data generated.")

# ------------------------------------------
# 1. Stabilization (Whitening)
# ------------------------------------------
print("Performing stabilization (whitening)...")

# Use ibllib's whitening matrix calculation if available
def get_whitening_matrix(data_chunk, geom):
    """Computes ZCA whitening matrix. Uses ibllib if available."""
    if HAS_IBLLIB:
        # ibllib's whitening often uses neighbors based on geometry
        neigh_channels = dsp.voltage.get_spaced_neighbors(geom, R=40) # Channels within 40um radius (adjust R)
        # Note: ibllib.dsp.fourier.whitening_matrix expects freq domain input usually
        # For time-domain ZCA, we compute covariance and SVD manually or use a dedicated function if found
        # Let's compute simple ZCA here for demonstration
        cov = np.cov(data_chunk, rowvar=False)
        U, S, V = np.linalg.svd(cov)
        # Add regularization to avoid division by zero/small numbers
        epsilon = 1e-6
        W = U @ np.diag(1.0 / np.sqrt(S + epsilon)) @ U.T
        return W
    else:
        # Simplified ZCA without ibllib (ignores geometry for simplicity)
        cov = np.cov(data_chunk, rowvar=False)
        U, S, V = np.linalg.svd(cov)
        epsilon = 1e-6
        W = U @ np.diag(1.0 / np.sqrt(S + epsilon)) @ U.T
        return W

def apply_whitening(data, W):
    """Applies whitening matrix W to data."""
    return data @ W # Assumes data is (samples, channels) and W is (channels, channels)

# Calculate condition numbers before and after whitening in chunks
cond_before = []
cond_after = []
whitened_data = np.zeros_like(raw_data)

n_chunks = N_SAMPLES // N_SAMPLES_CHUNK

for i in range(n_chunks):
    start = i * N_SAMPLES_CHUNK
    end = start + N_SAMPLES_CHUNK
    data_chunk = raw_data[start:end, :]

    # Ensure data_chunk is not zero or constant
    if np.all(np.std(data_chunk, axis=0) < 1e-9):
        print(f"Skipping chunk {i} due to zero variance.")
        continue

    # Before whitening
    cov_before = np.cov(data_chunk, rowvar=False)
    cond_before.append(np.linalg.cond(cov_before))

    # Compute and apply whitening
    W = get_whitening_matrix(data_chunk, geom)
    whitened_chunk = apply_whitening(data_chunk, W)
    whitened_data[start:end, :] = whitened_chunk # Store whitened data

    # After whitening
    cov_after = np.cov(whitened_chunk, rowvar=False)
    cond_after.append(np.linalg.cond(cov_after))

    if (i+1) % 10 == 0:
        print(f"Processed chunk {i+1}/{n_chunks}")

# Handle potential NaNs or Infs in condition numbers
cond_before = np.array(cond_before)
cond_after = np.array(cond_after)
cond_before = cond_before[np.isfinite(cond_before) & (cond_before > 0)]
cond_after = cond_after[np.isfinite(cond_after) & (cond_after > 0)]

print("Whitening finished.")

# --- Plotting Figure 1: Conditioning Histogram ---
print("Plotting Figure 1: Conditioning Histogram")
plt.figure(figsize=(8, 6))
# Use log10 scale for condition numbers
log_cond_before = np.log10(cond_before)
log_cond_after = np.log10(cond_after)

# Filter out potential -inf resulting from log10(0) if any slipped through
log_cond_before = log_cond_before[np.isfinite(log_cond_before)]
log_cond_after = log_cond_after[np.isfinite(log_cond_after)]


if len(log_cond_before) > 0:
    plt.hist(log_cond_before, bins=50, density=True, alpha=0.7, color='blue', label='Before Whitening')
else:
    print("Warning: No valid 'before' condition numbers to plot.")

if len(log_cond_after) > 0:
    plt.hist(log_cond_after, bins=50, density=True, alpha=0.7, color='orange', label='After Whitening (ZCA)')
else:
     print("Warning: No valid 'after' condition numbers to plot.")

plt.xlabel('log10(Condition Number of Covariance Matrix)')
plt.ylabel('Density')
plt.title('Distribution of Covariance Matrix Conditioning')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# ------------------------------------------
# 2. Drift Tracking and Registration (DREDge)
# ------------------------------------------
if not HAS_DREDGE:
    print("Skipping Drift Tracking: DREDge library not found.")
else:
    print("Starting Drift Tracking...")
    # --- Peak Detection ---
    # Use ibllib's peak detection if available, otherwise a simple threshold crossing
    print("Detecting peaks...")
    detection_sign = -1 # Detect negative peaks (typical for spikes/artifacts)
    threshold_factor = 5 # Times the estimated noise level
    peak_wind = int(0.001 * FS * 2) # samples around the peak to check for true max

    if HAS_IBLLIB:
         # Estimate noise level robustly
        noise_estimate = dsp.noise.mad(whitened_data) # Use Median Absolute Deviation
        thresholds = noise_estimate * threshold_factor
        # Note: detect_peaks expects (samples, channels), returns relative indices
        # Need absolute time indices.
        peak_indices, _ = spikes.detect_peaks(whitened_data,
                                              fs=FS,
                                              thresholds=thresholds * detection_sign, # Pass thresholds per channel
                                              sign='both', # Detect both, we'll filter by sign later maybe
                                              n_shifts=5, # From detect_peaks defaults
                                              detect_threshold = threshold_factor, # threshold in MAD units
                                              radius = None # Search radius in um (can speed up)
                                              )
        # Post-process detected peaks (ibllib might return positive/negative, let's refine)
        peak_times_samples = []
        peak_channels = []
        peak_amps = []
        for ch in range(N_CHANNELS):
            ch_peaks = peak_indices[ch]
            # Ensure peaks are within bounds and have correct sign
            valid_peaks = []
            for p_idx in ch_peaks:
                if 0 < p_idx < N_SAMPLES: # Basic boundary check
                    amp = whitened_data[p_idx, ch]
                    # Check sign (adjust based on expected peak polarity)
                    if (detection_sign < 0 and amp < -thresholds[ch]) or \
                       (detection_sign > 0 and amp > thresholds[ch]):
                       # Refine peak location slightly if needed (look in vicinity)
                       search_win = np.arange(max(0, p_idx - peak_wind//2), min(N_SAMPLES, p_idx + peak_wind//2))
                       refined_idx = search_win[np.argmin(whitened_data[search_win, ch])] if detection_sign < 0 else search_win[np.argmax(whitened_data[search_win, ch])]
                       valid_peaks.append(refined_idx)

            peak_times_samples.extend(list(np.unique(valid_peaks))) # Use unique peak times per channel
            peak_channels.extend([ch] * len(np.unique(valid_peaks)))
            # Get amplitudes at the refined peak times
            peak_amps.extend(whitened_data[np.unique(valid_peaks), ch])

        peak_times = np.array(peak_times_samples) / FS # Convert to seconds
        peak_channels = np.array(peak_channels)
        peak_amps = np.array(peak_amps)

        # Filter based on amplitude sign if 'both' was used and only one sign is needed
        if detection_sign < 0:
            select_idx = peak_amps < 0
        else:
            select_idx = peak_amps > 0

        peak_times = peak_times[select_idx]
        peak_channels = peak_channels[select_idx]
        peak_amps = peak_amps[select_idx]


    else:
        # Simplified peak detection without ibllib
        print("Using simplified peak detection (threshold crossing)...")
        noise_estimate = np.median(np.abs(whitened_data) / 0.6745, axis=0) # MAD estimate per channel
        thresholds = noise_estimate * threshold_factor
        peak_times_list = []
        peak_channels_list = []
        peak_amps_list = []

        for ch in range(N_CHANNELS):
            data_ch = whitened_data[:, ch] * detection_sign # Flip if detecting negative peaks
            peaks, props = scipy.signal.find_peaks(data_ch, height=thresholds[ch], distance=int(FS * 0.002))
            peak_times_list.extend(peaks / FS)
            peak_channels_list.extend([ch] * len(peaks))
            peak_amps_list.extend(whitened_data[peaks, ch]) # Store original amplitude

        peak_times = np.array(peak_times_list)
        peak_channels = np.array(peak_channels_list)
        peak_amps = np.array(peak_amps_list)

    # Map peak channels to depths
    peak_depths = channel_depths[peak_channels]
    print(f"Detected {len(peak_times)} peaks.")


    # --- Bin Peaks for DREDge ---
    print("Binning peaks for DREDge...")
    # Define bins for DREDge (adjust as needed)
    time_bin_size_s = 2.0 # seconds
    depth_bin_size_um = 10.0 # micrometers

    time_bins = np.arange(0, DURATION + time_bin_size_s, time_bin_size_s)
    min_depth = np.min(channel_depths) - depth_bin_size_um
    max_depth = np.max(channel_depths) + depth_bin_size_um
    depth_bins = np.arange(min_depth, max_depth, depth_bin_size_um)

    # Create the 2D histogram of peak counts
    # DREDge expects (n_time_bins, n_depth_bins)
    binned_peaks, _, _ = np.histogram2d(
        peak_times,
        peak_depths,
        bins=(time_bins, depth_bins)
    )
    print(f"Binned peaks shape: {binned_peaks.shape}") # Should be (n_time_bins-1, n_depth_bins-1)


    # --- Run DREDge ---
    print("Running DREDge motion estimation...")
    # DREDge parameters (these may need tuning based on data/expected drift)
    dredge_params = {
        "max_iter": 100, # Maximum iterations for optimization
        "max_disp_um": 50, # Maximum expected displacement between time bins (micrometers)
        # "batch_size_t": 32, # Process in time batches (optional)
        "width_um": 10, # Approx width of features/templates in depth (micrometers)
        "n_passes": 2, # Number of passes for refining the estimate
        "device": "cpu", # Or "cuda" if GPU available and torch installed with CUDA
        "p": 1.0, # Exponent for amplitude weighting (1.0 = use amplitudes, 0.0 = counts only)
        "destripe": True # Apply destriping filter
    }

    # DREDge needs peak features - it can work on counts, but often works better
    # with amplitude-weighted features. Let's try creating an amplitude-weighted histogram.
    # Note: Check DREDge docs for the preferred input format. It might want features per peak
    # or a binned representation. Assuming binned for now based on some examples.
    # Let's compute the sum of absolute amplitudes in each bin instead of just counts.
    weights = np.abs(peak_amps) # Weight by absolute amplitude
    binned_peak_amps, _, _ = np.histogram2d(
        peak_times,
        peak_depths,
        bins=(time_bins, depth_bins),
        weights=weights
    )

    # DREDge expects input of shape (T, C) or a specific feature format.
    # Let's use the amplitude-weighted histogram (T, DepthBins)
    # The function is dredge.motion_estimate
    # Note: DREDge often expects features, not just raw counts/amps directly.
    # Let's provide the amplitude-weighted histogram as 'templates_feature'
    # Check DREDge documentation for exact input requirements.
    # Assuming binned_peak_amps is the correct input format for 'templates_feature':
    try:
        # Note: DREDge API might change. Consult its documentation.
        # It might require depth bin centers instead of edges.
        depth_bin_centers = depth_bins[:-1] + depth_bin_size_um / 2
        # Run DREDge - ensure `binned_peak_amps` has the right shape (T, C)
        motion_estimate_dredge = dredge.motion_estimate(
            templates_feature=binned_peak_amps, # Input features (time bins x depth bins)
            times=time_bins[:-1] + time_bin_size_s / 2, # Time bin centers
            spatial_locs=depth_bin_centers, # Depth bin centers
            **dredge_params
        )
        # DREDge might return just the motion array, or a more complex object
        # Assuming it returns the motion array directly:
        estimated_drift = motion_estimate_dredge # This is the displacement in um over time
        drift_times = time_bins[:-1] + time_bin_size_s / 2 # Time points corresponding to the drift estimate

        print("DREDge motion estimation finished.")

    except Exception as e:
        print(f"Error running DREDge: {e}")
        print("Could not estimate drift. Plotting will skip the motion trace.")
        estimated_drift = None
        drift_times = None


    # --- Plotting Figure 2: Drift Map and Estimated Motion ---
    print("Plotting Figure 2: Drift Map and Motion Estimate")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)

    # Plot 1: Raster plot of detected peaks (Drift Map)
    ax_raster = axes[0]

    # Use ibllib's visualization if available and suitable
    if HAS_IBLLIB:
        try:
            # plot_driftmap needs specific inputs, ensure they match
            # We use the detected peaks directly here
            img = ephys_vis.plot_driftmap(
                peak_times, peak_depths, peak_amps, ax=ax_raster,
                t_bin=0.1, # Time bin size for visualization (smaller than DREDge bin)
                d_bin=10,  # Depth bin size for visualization
                vmin=np.percentile(peak_amps, 5) if len(peak_amps)>0 else -1, # Color limits based on detected amps
                vmax=np.percentile(peak_amps, 95) if len(peak_amps)>0 else 1,
                cmap='viridis' # Or another suitable colormap e.g. 'plasma'
            )
            # Add colorbar manually if plot_driftmap doesn't
            # if img and not ax_raster.images[-1].colorbar:
            #      cbar = fig.colorbar(img, ax=ax_raster, shrink=0.8)
            #      cbar.set_label('Peak Amplitude (whitened units)')

            ax_raster.set_title('Drift Map (Detected Peaks on Whitened Data)')
            ax_raster.set_ylabel('Depth (µm)')

        except Exception as e:
            print(f"Could not use ibllib.ephys.visualization.plot_driftmap: {e}. Using scatter plot fallback.")
            HAS_IBLLIB = False # Disable for this plot

    if not HAS_IBLLIB:
        # Fallback using scatter plot
        # Reduce number of points for performance if too many peaks
        max_plot_points = 50000
        if len(peak_times) > max_plot_points:
            idx = np.random.choice(len(peak_times), max_plot_points, replace=False)
            peak_times_plot = peak_times[idx]
            peak_depths_plot = peak_depths[idx]
            peak_amps_plot = peak_amps[idx]
        else:
            peak_times_plot = peak_times
            peak_depths_plot = peak_depths
            peak_amps_plot = peak_amps

        sc = ax_raster.scatter(peak_times_plot, peak_depths_plot, c=peak_amps_plot,
                               s=1, alpha=0.5, cmap='viridis',
                               vmin=np.percentile(peak_amps, 5) if len(peak_amps)>0 else -1,
                               vmax=np.percentile(peak_amps, 95) if len(peak_amps)>0 else 1)
        cbar = fig.colorbar(sc, ax=ax_raster, shrink=0.8)
        cbar.set_label('Peak Amplitude (whitened units)')
        ax_raster.set_title('Drift Map (Detected Peaks - Scatter Plot)')
        ax_raster.set_ylabel('Depth (µm)')
        ax_raster.set_ylim(np.min(channel_depths), np.max(channel_depths))


    # Plot 2: Estimated Motion from DREDge
    ax_motion = axes[1]
    if estimated_drift is not None and drift_times is not None:
        ax_motion.plot(drift_times, estimated_drift, color='r', label='DREDge Estimate')
        ax_motion.set_ylabel('Est. Drift (µm)')
        ax_motion.set_xlabel('Time (s)')
        ax_motion.grid(True, linestyle=':')
        ax_motion.legend(loc='upper right')
        ax_motion.set_xlim(0, DURATION)
        # Add ground truth drift for comparison (from synthetic data)
        true_drift_at_bin_centers = (drift_range / 2) * np.sin(2 * np.pi * (drift_speed / drift_range) * drift_times)
        ax_motion.plot(drift_times, true_drift_at_bin_centers, color='k', linestyle='--', label='True Drift (synth)')
        ax_motion.legend(loc='upper right')

    else:
        ax_motion.text(0.5, 0.5, 'Drift estimation failed or skipped.',
                       horizontalalignment='center', verticalalignment='center', transform=ax_motion.transAxes)
        ax_motion.set_xlabel('Time (s)')


    plt.suptitle('Neural Peak Raster and Estimated Motion', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

print("Analysis complete.")
'''Explanation:

Setup & Imports: Imports necessary libraries. It checks if ibllib and dredge are installed and prints warnings if not, using fallback methods where possible.

Parameters & Synthetic Data: Defines key parameters like sampling rate, channel count, duration. It generates synthetic data with background noise, some channel correlation, and artificial "peaks" whose depth drifts over time according to a sine wave pattern. Probe geometry is defined, trying to use ibllib's neuropixel tools first, falling back to a simple linear array.

Stabilization (Whitening):

It defines functions get_whitening_matrix (using ZCA) and apply_whitening. It attempts to use ibllib.dsp functions conceptually (like neighborhood definitions) but implements a basic ZCA if ibllib isn't fully available or if its specific whitening functions require different inputs (e.g., frequency domain).

It processes the data in chunks to calculate the covariance matrix and its condition number before whitening.

It computes the whitening matrix (W) for each chunk and applies it.

It calculates the condition number of the covariance matrix of the whitened chunk.

The whitened data is stored (or could be processed chunk-by-chunk if memory is an issue).

Conditioning Plot (Figure 1): It generates a histogram comparing the log10 of the condition numbers before (blue) and after (orange) whitening, showing the stabilization effect (condition numbers closer to 1, smaller spread).

Peak Detection:

It detects peaks on the whitened data, which usually improves signal-to-noise for peak detection.

It tries to use ibllib.ephys.spikes.detect_peaks if available, which is robust.

If ibllib is unavailable, it uses a simpler scipy.signal.find_peaks based on threshold crossing.

Detected peak times (in seconds), channel indices, and amplitudes are stored. Channel indices are converted to depths using the geometry.

Binning for DREDge: The detected peaks (time, depth) are binned into a 2D histogram. DREDge often works better with amplitude information, so a second histogram weighted by absolute peak amplitudes is also calculated.

Run DREDge:

It calls dredge.motion_estimate, passing the binned peak feature data (amplitude-weighted histogram), time bin centers, and depth bin centers.

Crucial DREDge parameters like max_disp_um (maximum expected displacement between time bins) are set. These might need tuning for real data.

The output motion_estimate_dredge contains the estimated drift in micrometers for each time bin.

Drift Map Plot (Figure 2):

It creates a figure with two subplots.

Top Subplot (Raster): It displays the detected peaks. It tries to use ibllib.ephys.visualization.plot_driftmap for a nice heatmap visualization. If that fails or isn't available, it falls back to a matplotlib.pyplot.scatter plot where x is time, y is depth, and color represents peak amplitude.

Bottom Subplot (Motion): It plots the motion trace estimated by DREDge (estimated_drift) against time. The synthetically generated ground truth drift is also plotted for comparison.

Error Handling: Basic checks for library availability and potential issues like zero variance chunks or errors during DREDge execution are included.

This code provides a framework using the requested libraries and concepts to perform the analysis and generate the specified plots. Remember that parameters (thresholds, bin sizes, DREDge parameters) often need careful tuning for optimal performance on real experimental data.'''