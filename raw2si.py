import time, os, json, warnings, shutil
import probeinterface as pi
from probeinterface.plotting import plot_probe
import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.postprocessing as spost
import spikeinterface.sorters as ss
import spikeinterface.qualitymetrics as sq
import spikeinterface.exporters as sep

# from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
# from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording
from spikeinterface.sortingcomponents.motion import (
    correct_motion_on_peaks,
    interpolate_motion,
)
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import numcodecs

##from brainbox.population.decode import get_spike_counts_in_bins
##from brainbox.task.trials import get_event_aligned_raster, get_psth
##from brainbox.ephys_plots import scatter_raster_plot
warnings.simplefilter("ignore")
import spikeinterface.curation as scur

"""
This pipeline uses spikeinterface as a backbone. This file includes preprocessing and sorting, converting raw data from openEphys to putative spikes by various sorters
"""


def generate_sorter_suffix(this_sorter):
    if this_sorter.lower() == "spykingcircus2":
        sorter_suffix = "_SC2"
    elif this_sorter.lower() == "kilosort3":
        sorter_suffix = "_KS3"
    elif this_sorter.lower() == "kilosort4":
        sorter_suffix = "_KS4"
    return sorter_suffix


def raw2si(thisDir, json_file):
    oe_folder = Path(thisDir)
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    this_sorter = analysis_methods.get("sorter_name")
    this_experimenter = analysis_methods.get("experimenter")
    probe_type = analysis_methods.get("probe_type")
    motion_corrector = analysis_methods.get("motion_corrector")
    load_existing_motion_info = True
    sorter_suffix = generate_sorter_suffix(this_sorter)
    result_folder_name = "results" + sorter_suffix
    sorting_folder_name = "sorting" + sorter_suffix
    n_cpus = os.cpu_count()
    n_jobs = n_cpus - 4
    job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)

    if analysis_methods.get("load_sorting_file") == True:
        if (oe_folder / result_folder_name).is_dir():
            # sorting_spikes = ss.read_sorter_folder(oe_folder/result_folder_name)
            sorting_spikes = si.load_extractor(
                oe_folder / sorting_folder_name
            )  # this acts quite similar than above one line.
        else:
            print(f"no result folder found for {this_sorter}")
        # load recording in case there is a need to extract waveform
        if (oe_folder / "preprocessed_compressed.zarr").is_dir():
            recording_saved = si.read_zarr(oe_folder / "preprocessed_compressed.zarr")
            print(recording_saved.get_property_keys())
        elif (oe_folder / "preprocessed").is_dir():
            recording_saved = si.load_extractor(oe_folder / "preprocessed")
        else:
            print(f"no pre-processed folder found. Unable to extract waveform")
            return sorting_spikes
        recording_saved.annotate(is_filtered=True)
    else:
        if (
            analysis_methods.get("load_prepocessed_file") == True
            and (oe_folder / "preprocessed_compressed.zarr").is_dir()
        ):
            recording_saved = si.read_zarr(oe_folder / "preprocessed_compressed.zarr")
            print(recording_saved.get_property_keys())
            fs = recording_saved.get_sampling_frequency()
        elif (
            analysis_methods.get("load_prepocessed_file") == True
            and (oe_folder / "preprocessed").is_dir()
        ):
            print(
                "Looks like you do not have compressed files. Read the original instead"
            )
            recording_saved = si.load_extractor(oe_folder / "preprocessed")
            fs = recording_saved.get_sampling_frequency()
        else:
            print("Load meta information from openEphys")
            full_raw_rec = se.read_openephys(oe_folder, load_sync_timestamps=True)
            # To show the start of recording time
            # full_raw_rec.get_times()[0]
            event = se.read_openephys_event(oe_folder)
            # event_channel_ids=channel_ids
            # events = event.get_events(channel_id=channel_ids[1], segment_index=0)# a complete record of events including [('time', '<f8'), ('duration', '<f8'), ('label', '<U100')]
            events_times = event.get_event_times(
                channel_id=event.channel_ids[1], segment_index=0
            )  # this record ON phase of sync pulse
            fs = full_raw_rec.get_sampling_frequency()
            if analysis_methods.get("load_raw_traces") == True:
                trace_snippet = full_raw_rec.get_traces(
                    start_frame=int(fs * 0), end_frame=int(fs * 2)
                )

            ################load probe information################
            if probe_type == "P2":
                manufacturer = "cambridgeneurotech"
                probe_name = "ASSY-37-P-2"
                probe = pi.get_probe(manufacturer, probe_name)
                print(probe)
                probe.wiring_to_device("ASSY-116>RHD2132")
                probe.to_dataframe(complete=True).loc[
                    :, ["contact_ids", "shank_ids", "device_channel_indices"]
                ]
            elif probe_type == "H10_stacked":
                stacked_probes = pi.read_probeinterface("H10_stacked_probes.json")
                probe = stacked_probes.probes[0]
            else:
                print("the name of probe not identified. stop the programme")
                return
            # drop AUX channels here
            raw_rec = full_raw_rec.set_probe(probe)
            probe_rec = raw_rec.get_probe()
            probe_rec.to_dataframe(complete=True).loc[
                :, ["contact_ids", "device_channel_indices"]
            ]

            raw_rec.annotate(
                description=f"Dataset of {this_experimenter}"
            )  # should change here for something related in the future
            ################ preprocessing ################
            # apply band pass filter
            recording_f = spre.bandpass_filter(raw_rec, freq_min=600, freq_max=6000)
            # apply common median reference to remove common noise
            if analysis_methods.get("analyse_good_channels_only") == True:
                """
                This step should be done before saving preprocessed files because ideally the preprocessed file we want to create is something ready for spiking
                detection, which means neural traces gone through bandpass filter and common reference.
                However, applying common reference takes signals from channels of interest which requires us to decide what we want to do with other bad or noisy channels first.
                """
                bad_channel_ids, channel_labels = spre.detect_bad_channels(
                    recording_f, method="coherence+psd"
                )  # bad_channel_ids=np.array(['CH1','CH2','CH3','CH4','CH5','CH6','CH7','CH8','CH9','CH10','CH11','CH12','CH13','CH14','CH15','CH16'],dtype='<U64')
                print("bad_channel_ids", bad_channel_ids)
                print("channel_labels", channel_labels)

                recording_f = recording_f.remove_channels(
                    bad_channel_ids
                )  # need to check if I can do this online

            recording_cmr = spre.common_reference(
                recording_f, reference="global", operator="median"
            )
        if "recording_cmr" in locals():
            rec_of_interest = recording_cmr
        else:
            rec_of_interest = recording_saved
            rec_of_interest.annotate(
                is_filtered=True
            )  # needed to add this somehow because when loading a preprocessed data saved in the past, that data would not be labeled as filtered data
        # Slice the recording if needed
        if analysis_methods.get("analyse_entire_recording") == False:
            start_sec = 1
            end_sec = 899
            rec_of_interest = rec_of_interest.frame_slice(
                start_frame=start_sec * fs, end_frame=end_sec * fs
            )
        # at the moment, leaving raw data intact while saving preprocessed files in compressed format but in the future,
        # we might want to remove the raw data to save space
        # more information about this idea can be found here https://github.com/SpikeInterface/spikeinterface/issues/2996#issuecomment-2486394230
        compressor_name = "zstd"
        if (oe_folder / "preprocessed_compressed.zarr").is_dir():
            if (
                analysis_methods.get("save_prepocessed_file") == True
                and analysis_methods.get("overwrite_curated_dataset") == True
            ):
                compressor = numcodecs.Blosc(
                    cname="zstd", clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE
                )
                recording_saved = rec_of_interest.save(
                    format="zarr",
                    folder=oe_folder / "preprocessed_compressed.zarr",
                    compressor=compressor,
                    overwrite=True,
                    **job_kwargs,
                )
                print(f"Overwrite existing file with compressor: {compressor_name}")
            else:
                recording_saved = rec_of_interest
                print(
                    "Skip saving this Recording object and converting it to a binary file. Please make sure your sorters are happy with that"
                )
        else:
            compressor = numcodecs.Blosc(
                cname=compressor_name, clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE
            )
            recording_saved = rec_of_interest.save(
                format="zarr",
                folder=oe_folder / "preprocessed_compressed.zarr",
                compressor=compressor,
                **job_kwargs,
            )
            print(
                f"First time to save this file. Testing compressor: {compressor_name}"
            )

        ############################# correcting drift/motion ##########################
        # There are multiple ways to correct motion artifact. We can either use kilosort's motion correction or other methods like dredge.
        # kilosort's motion correction is part of its own preprocessing step after removing common noise with common median reference (CMR) or average reference (CAR).
        # Therefore, there are drift correction and whitening steps if kilosort or kilosort's motion correction is not used.
        # Motion correction method Dredge seems do well with shorter probe https://www.youtube.com/watch?v=RYKHoipT-2A
        # Note: if we use kilosort without using its own preprocessing steps, remember to turn off those steps when calling the function.
        # For more information, please refer to https://github.com/SpikeInterface/spikeinterface/issues/3483

        if motion_corrector == ("dredge"):
            motion_folder = oe_folder / "motion"
            # dredge_preset_params = spre.get_motion_parameters_preset("dredge")
            if motion_folder.exists() and load_existing_motion_info:
                motion_info = spre.load_motion_info(motion_folder)
                recording_corrected = interpolate_motion(
                    recording=recording_saved,
                    motion=motion_info["motion"],
                    temporal_bins=motion_info["temporal_bins"],
                    spatial_bins=motion_info["spatial_bins"],
                )
            else:
                win_um = 75
                recording_corrected, _, motion_info = spre.correct_motion(
                    recording=recording_saved,
                    preset=preset,
                    folder=motion_folder,
                    overwrite=False,
                    output_motion=True,
                    output_motion_info=True,
                    estimate_motion_kwargs={
                        "win_step_um": win_um,
                        "win_scale_um": win_um,
                    },
                    **job_kwargs,
                )  # the default mode will remove channels at the border, trying using force_extrapolate
        elif motion_corrector == ("kilosort"):
            use_kilosort_motion_correction = True
            rec_for_sorting = recording_saved
        elif motion_corrector == ("testing"):
            ## this is a section to test which algorithm is better for motion correction. This is based on this page https://spikeinterface.readthedocs.io/en/latest/how_to/handle_drift.html
            some_presets = (
                "rigid_fast",
                "kilosort_like",
                "nonrigid_accurate",
                "nonrigid_fast_and_accurate",
                "dredge",
                "dredge_fast",
            )

            run_times = []

            win_um = 125
            for preset in some_presets:
                print("Computing with", preset)
                test_folder = oe_folder / f"motion_folder_dataset{win_um}" / preset
                if load_existing_motion_info and test_folder.exists():
                    motion_info = spre.load_motion_info(test_folder)
                else:
                    recording_corrected, _, motion_info = spre.correct_motion(
                        recording=recording_saved,
                        preset=preset,
                        folder=test_folder,
                        overwrite=False,
                        output_motion=True,
                        output_motion_info=True,
                        estimate_motion_kwargs={
                            "win_step_um": win_um,
                            "win_scale_um": win_um,
                        },
                        **job_kwargs,
                    )  # the default mode will remove channels at the border, trying using force_extrapolate
                    fig = plt.figure(figsize=(14, 8))
                    sw.plot_motion_info(
                        motion_info,
                        recording_corrected,
                        figure=fig,
                        depth_lim=(0, 400),
                        color_amplitude=True,
                        amplitude_cmap="inferno",
                        scatter_decimate=10,
                    )
                    fig.suptitle(f"{preset=}")
                    fig.savefig(test_folder / "estimated_motion_result.png")
                run_times.append(motion_info["run_times"])
                """this part is not yet useful because it does not seem that the motion is estimated and corrected  correctly in 3D
                #fig2, axs = plt.subplots(ncols=2, figsize=(12, 8), sharey=True)
                fig2=plt.figure()
                ax=fig2.add_subplot(121,projection='3d')
                #ax = axs[0]
                #sw.plot_probe_map(recording_corrected, ax=ax)

                peaks = motion_info["peaks"]
                time_lim0 = 0.0#750.0
                time_lim1 = 1000.0#1500.0
                mask = (peaks["sample_index"] > int(fs * time_lim0)) & (peaks["sample_index"] < int(fs * time_lim1))
                sl = slice(None, None, 5)
                amps = np.abs(peaks["amplitude"][mask][sl])
                amps /= np.quantile(amps, 0.95)
                c = plt.get_cmap("inferno")(amps)

                color_kargs = dict(alpha=0.2, s=2, c=c)

                peak_locations = motion_info["peak_locations"]
                # color='black',
                ax.scatter(peak_locations["x"][mask][sl], peak_locations["y"][mask][sl],peak_locations["z"][mask][sl], **color_kargs)
                ax.set_ylim(0, 400)
                peak_locations2 = correct_motion_on_peaks(peaks, peak_locations, motion,recording_saved)
                
                ax=fig2.add_subplot(122,projection='3d')
                #ax = axs[1]
                #sw.plot_probe_map(recording_saved, ax=ax)
                #  color='black',
                ax.scatter(peak_locations2["x"][mask][sl], peak_locations2["y"][mask][sl],peak_locations2["z"][mask][sl],**color_kargs)

                ax.set_ylim(0, 400)
                fig2.suptitle(f"{preset=}")
                fig2.savefig(test_folder/'estimated_motion_location.png')'
                """

            keys = run_times[0].keys()

            bottom = np.zeros(len(run_times))
            fig3, ax = plt.subplots(figsize=(14, 6))
            for k in keys:
                rtimes = np.array([rt[k] for rt in run_times])
                if np.any(rtimes > 0.0):
                    ax.bar(some_presets, rtimes, bottom=bottom, label=k)
                bottom += rtimes
            ax.legend()
            fig3.savefig(
                oe_folder
                / f"motion_folder_dataset{win_um}"
                / "run_time_accuracy_comparsion.png"
            )
            return
        else:
            print(
                "input name of motion corrector not identified so do not correct motion/drift"
            )
            recording_corrected = recording_saved
        ############################# whitening ##########################
        if use_kilosort_motion_correction:
            print("use the default whitening method in the kilosort")
            pass
        else:
            # apply whitening to remove spatial correction in the data
            step_chan = 25
            rec_for_sorting = spre.whiten(
                recording=recording_corrected,
                mode="local",
                radius_um=step_chan * 2,
                dtype=float,
            )

        ############################# spike sorting ##########################
        sorter_params = si.get_default_analyzer_extension_params("this_sorter")
        if this_sorter.startswith("kilosort"):
            if use_kilosort_motion_correction:
                pass
            else:
                sorter_params.update({"skip_kilosort_preprocessing": True})
            if this_sorter == "kilosort3":
                kilosort_3_path = r"C:\Users\neuroPC\Documents\GitHub\Kilosort-3.0.2"
                ss.Kilosort3Sorter.set_kilosort3_path(kilosort_3_path)
                sorter_params = {"do_correction": False}
            else:
                print("use kilosort4")
                # sorter_params={'dminx': 250,'nearest_templates':10}
            sorting_spikes = ss.run_sorter(
                sorter_name=this_sorter,
                recording=rec_for_sorting,
                remove_existing_folder=True,
                do_CAR=False,
                output_folder=oe_folder / result_folder_name,
                verbose=True,
                **sorter_params,
            )
            # sorter_params.update({"projection_threshold": [9, 9]})##this is a parameters from Christopher Michael Jernigan's experiences with Wasps
        else:
            sorter_params = ss.get_default_sorter_params(this_sorter)
            print(f"run spike sorting with {this_sorter}")
            sorting_spikes = ss.run_sorter(
                sorter_name=this_sorter,
                recording=rec_for_sorting,
                remove_existing_folder=True,
                output_folder=oe_folder / result_folder_name,
                verbose=True,
                job_kwargs=job_kwargs,
            )
        ##this will return a sorting object
    ############################# spike sorting preview and saving ##########################
    w_rs = sw.plot_rasters(sorting_spikes, time_range=(0, 30), backend="matplotlib")
    if (
        analysis_methods.get("save_sorting_file") == True
        and analysis_methods.get("overwrite_curated_dataset") == True
    ):
        sorting_spikes.save(folder=oe_folder / sorting_folder_name, overwrite=True)

    return print("Spiking sorting done. The rest of the tasks can be done in other PCs")
    # for unit in sorting_spikes.get_unit_ids():
    #     print(f'with {this_sorter} sorter, Spike train of a unit:{sorting_spikes.get_unit_spike_train(unit_id=unit)}')


if __name__ == "__main__":
    # thisDir = r"C:\Users\neuroLaptop\Documents\Open Ephys\P-series-32channels\GN00003\2023-12-28_14-39-40"
    # thisDir = r"Z:\DATA\experiment_openEphys\P-series-32channels\2024-02-01_15-25-25"
    # thisDir = r"C:\Users\neuroPC\Documents\Open Ephys\2024-05-01_17-39-51"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23015\240201\coherence\session1\2024-02-01_15-25-25"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23016\240201\coherence\session1\2024-02-01_18-55-51"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN24001\240529\coherence\session1\2024-05-29_15-33-31"
    # thisDir = r"D:\Open Ephys\2025-03-10_20-25-05"
    thisDir = r"D:\Open Ephys\2025-03-19_18-02-13"
    # thisDir = r"D:\Open Ephys\2025-02-23_20-39-04"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23019\240507\coherence\session1\2024-05-07_23-08-55"
    # thisDir = r"Z:\DATA\experiment_trackball_Optomotor\Zball\GN23018\240422\coherence\session2\2024-04-22_01-09-50"
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    raw2si(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")
