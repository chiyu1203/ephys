import time, os, json, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.core as si
import spikeinterface.extractors as se
import probeinterface as pi
from probeinterface.plotting import plot_probe
import spikeinterface.preprocessing as spre
import spikeinterface.widgets as sw
from spikeinterface.sortingcomponents.motion import (
    correct_motion_on_peaks,
    interpolate_motion,
    estimate_motion,
)
n_cpus = os.cpu_count()
n_jobs = n_cpus - 4
global_job_kwargs = dict(n_jobs=n_jobs, chunk_duration="2s")
si.set_global_job_kwargs(**global_job_kwargs)
def LFP_band_drift_estimation(group,raw_rec,oe_folder):
    lfprec = spre.bandpass_filter(raw_rec,freq_min=0.5,freq_max=250,margin_ms=1500.,filter_order=3,dtype="float32",add_reflect_padding=True)
    lfprec = spre.common_reference(lfprec,reference="global", operator="median")
    lfprec = spre.resample(lfprec, resample_rate=250, margin_ms=1000)
    lfprec = spre.average_across_direction(lfprec)
    fig0=plt.figure()
    ax=fig0.add_subplot(121)
    sw.plot_traces(lfprec, backend="matplotlib", mode="auto",ax=ax,time_range=(0, 1))
    #sw.plot_traces(lfprec, backend="matplotlib", mode="auto", ax=ax, clim=(-0.05, 0.05),time_range=(0, 20))
    motion_lfp = estimate_motion(lfprec, method='dredge_lfp', rigid=True, progress_bar=True)
    ax=fig0.add_subplot(122)
    sw.plot_motion(motion_lfp, mode='line', ax=ax)
    motion_folder = oe_folder / f"lfp_motion_shank{group}"
    if Path(motion_folder).is_dir():
        pass
    else:
        motion_folder.mkdir(parents=True, exist_ok=True)
    fig0.savefig(motion_folder / "dredge_lfp.png")
    plt.show()
    return motion_lfp


def AP_band_drift_estimation(group,recording_saved,oe_folder,analysis_methods,win_step_um,win_scale_um):
    load_existing_motion_info=analysis_methods.get("load_existing_motion_info")
    motion_corrector = analysis_methods.get("motion_corrector")
    skip_motion_correction=analysis_methods.get("skip_motion_correction")
    motion_folder = oe_folder / f"motion_shank{group}"
    motion_info_list=[]
    motion_corrector_tuple=("dredge","rigid_fast","kilosort_like")
    #motion_corrector_tuple=("dredge","kilosort_like")
    if skip_motion_correction:
        print(
            "skipp correct motion/drift"
        )
        recording_corrected = recording_saved
    else:
        if motion_corrector in motion_corrector_tuple:
            test_folder = motion_folder / motion_corrector
            motion_corrector_params = spre.get_motion_parameters_preset(motion_corrector)
            motion_corrector_params['estimate_motion_kwargs'].update({"win_step_um":75.0,"win_scale_um":250.0,"win_margin_um":150})
            # dredge_preset_params = spre.get_motion_parameters_preset("dredge")
            if test_folder.is_dir() and load_existing_motion_info:
                motion_info = spre.load_motion_info(test_folder)
                recording_corrected = interpolate_motion(
                    recording=recording_saved,
                    motion=motion_info["motion"],
                    #temporal_bins=motion_info["temporal_bins"],
                    #spatial_bins=motion_info["spatial_bins"],
                )
            elif analysis_methods.get("overwrite_curated_dataset") or test_folder.is_dir()==False:
                # if motion_corrector == "kilosort_like":
                #     estimate_motion_kwargs = {
                # elif motion_corrector == "dredge":
                recording_corrected, _, motion_info = spre.correct_motion(
                    recording=recording_saved,
                    preset=motion_corrector,
                    folder=test_folder,
                    overwrite=True,
                    output_motion=True,
                    output_motion_info=True,
                    estimate_motion_kwargs=motion_corrector_params['estimate_motion_kwargs']
                )
            else:
                motion_info=[]
                recording_corrected, _, motion_info = spre.correct_motion(
                    recording=recording_saved,
                    preset=motion_corrector,
                    folder=test_folder,
                    overwrite=False,
                    output_motion=False,
                    output_motion_info=False,
                    estimate_motion_kwargs=motion_corrector_params['estimate_motion_kwargs'])#interpolate_motion_kwargs={'border_mode' : 'force_extrapolate'},
                print('recording is corrected but output_motion and info are not generated')
            motion_info_list.append(motion_info)  # the default mode will remove channels at the border, trying using force_extrapolate
        elif motion_corrector == ("testing"):
            # This is a section to test which algorithm is better for motion correction. 
            # This is based on this page https://spikeinterface.readthedocs.io/en/latest/how_to/handle_drift.html

            run_times = []
            for preset in motion_corrector_tuple:
                print("Computing with", preset)
                if preset == "rigid_fast":
                    test_folder = oe_folder / f"motion_shank{group}_dataset" / preset
                else:
                    test_folder = oe_folder / f"motion_shank{group}_dataset{win_step_um}_{win_scale_um}" / preset
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
                            "win_step_um": win_step_um,
                            "win_scale_um": win_scale_um,
                            #"win_margin_um": win_margin_um,
                        })  # the default mode will remove channels at the border, trying using force_extrapolate
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
                motion_info_list.append(motion_info)
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
                    ax.bar(motion_corrector_tuple, rtimes, bottom=bottom, label=k)
                bottom += rtimes
            ax.legend()
            fig3.savefig(oe_folder / f"motion_shank{group}_dataset{win_step_um}_{win_scale_um}" / "run_time_accuracy_comparsion.png")
            recording_corrected = recording_saved
            plt.close('all')
        else:
            print(
                "input name of motion corrector not identified so do not correct motion/drift"
            )
            recording_corrected = recording_saved
    return recording_corrected,motion_info_list

def run_estimation(thisDir, json_file):
    oe_folder = Path(thisDir)
    if isinstance(json_file, dict):
        analysis_methods = json_file
    else:
        with open(json_file, "r") as f:
            print(f"load analysis methods from file {json_file}")
            analysis_methods = json.loads(f.read())
    this_experimenter = analysis_methods.get("experimenter")
    probe_type = analysis_methods.get("probe_type")
    motion_corrector = analysis_methods.get("motion_corrector")
    lfp_drift_estimation=False
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

        raw_rec = se.read_openephys(oe_folder, load_sync_timestamps=True)
        # To show the start of recording time
        # raw_rec.get_times()[0]
        event = se.read_openephys_event(oe_folder)
        # event_channel_ids=channel_ids
        # events = event.get_events(channel_id=channel_ids[1], segment_index=0)# a complete record of events including [('time', '<f8'), ('duration', '<f8'), ('label', '<U100')]
        events_times = event.get_event_times(
            channel_id=event.channel_ids[1], segment_index=0
        )  # this record ON phase of sync pulse
        fs = raw_rec.get_sampling_frequency()
        if analysis_methods.get("load_raw_traces") == True:
            trace_snippet = raw_rec.get_traces(
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
        raw_rec = raw_rec.set_probe(probe,group_mode='by_shank')
        probe_rec = raw_rec.get_probe()
        probe_rec.to_dataframe(complete=True).loc[
            :, ["contact_ids", "device_channel_indices"]
        ]

        raw_rec.annotate(
            description=f"Dataset of {this_experimenter}"
        )

        raw_rec_dict = raw_rec.split_by(property='group', outputs='dict')
        if lfp_drift_estimation:
            motion_lfp_dict={}
            for group, rec_per_shank in raw_rec_dict.items():
                motion_lfp=LFP_band_drift_estimation(group,rec_per_shank,oe_folder)
                motion_lfp_dict[group]=motion_lfp


        
    recordings_dict = recording_saved.split_by(property='group', outputs='dict')
    win_step_set=[75,50,25]
    #win_scale_set=[150,200,250]
    win_scale_set=[250,200,150]
    #win_margin_set=[-150,0,150]
    #win_margin_set=[150,0]
    # win_step_um":75.0,"
    # win_scale_um":250.0
    #win_step_set=[50,100,150]
    #win_scale_set=[50,100,150,200,250]
    recording_corrected_dict = {}
    motion_ap_dict={}
    for group, sub_recording in recordings_dict.items():
        if group==1:#if group%2==1:
            continue
        for win_scale_um in win_scale_set:
            for win_step_um in win_step_set:
                #for win_margin_um in win_margin_set:
                print(f"win_step_um={win_step_um}, win_scale_um={win_scale_um}")#,win_margin_um={win_margin_um}
                _,_=AP_band_drift_estimation(group,sub_recording,oe_folder,analysis_methods,win_step_um,win_scale_um)
        # recording_corrected_dict[group]=recording_corrected
        # motion_ap_dict[group]=motion_ap_list
        
    return print("done testing motion correction")



if __name__ == "__main__":
    #thisDir = r"D:\Open Ephys\2025-03-10_20-25-05"
    #thisDir = r"D:\Open Ephys\2025-03-19_18-02-13"
    #thisDir = r"Z:\DATA\experiment_openEphys\H-series-128channels\2025-03-23_20-47-26"
    #thisDir = r"Z:\DATA\experiment_openEphys\H-series-128channels\2025-04-09_22-46-23"
    thisDir = r"Z:\DATA\experiment_openEphys\H-series-128channels\2025-04-09_21-22-00"
    #thisDir = r"Z:\DATA\experiment_openEphys\H-series-128channels\2025-03-23_21-33-38"
    #thisDir = r"Z:\DATA\experiment_openEphys\H-series-128channels\2025-03-23_20-47-26"
    json_file = "./analysis_methods_dictionary.json"
    ##Time the function
    tic = time.perf_counter()
    run_estimation(thisDir, json_file)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to run the main function")