### ephys
General anaconda environment for all setups and data analyses

Install the latest anaconda version for your operating system (https://www.anaconda.com/products/individual).

Open the anaconda prompt and create a virtual environment via conda. Note: both ibllib and kilosort4 have started to move to numpy 2.0 already. It seems that Spikeinterface is compatible with them but not sure yet.

````
conda create --name ephys python=3.12
conda activate ephys
````
Then installing spikeinterface ibllib and other dependencies. Then installing other dependencies. (open ephys python tool is for loading timestamp; zarr and numcodesc for compressing data; ipympl is for interactive plots on Jupyter notebook. Pyside 6 is for spikeinterface-gui )

````
pip install ibllib spikeinterface[full,widgets] zarr docker cuda-python numcodecs hdbscan ipympl spikeinterface-gui PySide6
````

[optional] ibllib has many plotting functions You can either install it via pip or to install it from source. Below is the version I forked from the source

Note: ibllib use scipy 1.12 but installing one of these packages **zarr docker cuda-python numcodecs hdbscan** 
needs scipy 1.13 so I hopes there is no conflict between them 

````
pip install git+https://github.com/chiyu1203/ibllib.git
````

If you have a good GPU and wants to install kilosort. Here is the instruction under python 3.11.
````
python -m pip install kilosort[gui]=4.1.1
pip uninstall torch
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
#conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
````
Note: kilosort updates frequently so that spikeinterface can not always catch up with that. Therefore, I kept a seperate virtual environment dedicated for kilosort in case I want to use it standalone or use its GUI.

Probably dont need the packages below anymore
````
conda install --yes -c conda-forge -v ipython jupyter pytest
````

If you want to use [phy](https://github.com/cortex-lab/phy) for manual curation of spike sorting, create a seperate virtual environment and install it via this command
````
conda create -n phy2 -y python=3.11 cython dask h5py joblib matplotlib numpy pillow pip pyopengl pyqt pyqtwebengine pytest python qtconsole requests responses scikit-learn scipy traitlets
````

When using phy for curation, go to the directory where phy mata file is stored and type the following:

````
conda activate phy2
cd path/to/my/spikesorting/output
phy template-gui params.py
````
**e.g. D:\Open Ephys\2025-02-23_20-39-04\phy_KS4>phy template-gui params.py**

# data collection (doing a 3 hour recording with a 32-channel probe generates data at around 25 GB, double check if neuroPC has enough capacity before starting a session, a 80-min recording with 128 channels results in 40 GB)

# Setting up Surround displays
Surround displays are used to create panorama view. To set up surround displays, go to NVIDIA Control Panel with all 4 displays on and click Configure Surround, PhysX. Then click Configure, where you can specify which display to use and their relative position. In the case of 5840 * 1080 bezel corrected resolution, click on 5760 *1080 first resolution first and then add 40 Bezels value to each border (V1, V2). Note: it is possible to set the Refresh Rate to 240 Hz, however, we dont have/need that much fast camera and our computer has the closed-loop VR system to work on already so why bother.
After that, enable Surround to let the effect kick in. Usually this will result the 4th display to turn off. To reconnect the 4th display, go to Set up multiple displays on the menus and click up the 4th display. Lastly, go to Change resolution to double check the resolution is  5840 * 1080 bezel corrected and 144 Hz refresh rate. (if ever needed, go to Set up G-SYNC to enable G-SYNC, G-SYNC Compatible). In general, this only needs to be setup once. However, the surround displays can break when turning on the computer with the surround displays stays "off" or any other reason. As long as you see Bonsai open the windows in the 4th display. That is usually the case when the surround display is off. Use calibrate display color to do gamma calibration (max out the value for distortion, which would reduce the contrast of the monitor). Note: gamma calibration will break if only one monitor is on (not all) when starting the computer or waking up the computer.

# Linlab 2 
If there is any problem to load Linlab or connect Linlab with the micromanipulator, restart the computer.


# Probe mapping
We use probes from CambridgeNeurotech. The mapping can be found via probeinterface, for more [details](https://github.com/SpikeInterface/probeinterface/issues/301), or customised adjusted json or prb file in the repository. For example, 'H10_RHD2164.prb' or 'H10_RHD2164_rev.prb'. rev means the intan chip is inserted in the opposite direction.

# Preparation

1. Turn on air conditioning and PC to heat up the room in the morning [Optional] Use dehumilifier to reduce the humidity overnight.

2. Prepare locust saline, dyes, tin foil (to wrap the dye), wax, parafilm and a beaker with trypsin for electrode, a beaker for locust saline, 1 pipettes (for loading dye) and 1 pipetteman, distill water and isopropyl alcohol. Place those reagents on ice in a box except for locust saline. Then rinse the probe with isopropyl alcohol to clear the remaining dye (followed by rinsing with distilled water).

3. place the locust on ice for 5 mins and then fix the locust with wax on the customised headbar. Make an insertion site on the exoskeleton. The site should be in between the antenna (and somewhere ventral to the antenna). Then slide it either in between the antenna if you want to preserve them or cut off the antenna nerve to remove the antenna.

4. Remove airsacs, cuticle, and fat body until you can see the brain. [Optional] cut off the gut: to do that, it is actually better to cut the face all the way until mouth so that there are space to find the gut (have not decided whether I should remove the gut or not). Install one or two metal bars to stablise the brain. Insert the reference pin. Fix the wound with wax (the ventral part of the head; for the dorsal part, just use insersion site to stablise them) and then remove the neural sheath.
  
Note: The downside of removing the gut is that the head would become a big dry hole so I was not sure if putting the ground pin at the dorsal side of the head would work. One idea is to stuff moist kimwipe beneath the brain. Or maybe buy dura-gel would be a good idea. However, keep in mind during the surgery that if the brain does not move too much due to animal's breath, it should be possible to do the recording without removing the gut

5. When inserting probe from drosal side, trying removing the neural sheath from the dorsal side of the central complex and ripe the neural sheet along DV axis. This is because the probe is inserted along along DV axis. When inserting probe from anterior side, try removing the neural sheath toward the ipsilateral side (the recording site) so that tissue there is not pulled too much

6. Place the locust on the airball and then prepare to stain the electrode. (Remember to connect the SPL wire with the intan chip before putting them on the stereotaxis)

7. turn on LinLab2, micromanipulators(AP axis = x, + means anterior; LM axis = y, + means lateral, DV axis = z, + mean ventral), calibrate monitor colour with Calibrate Display Color app. 

8. Once the probe is around the potential location. Connect it the reference wire (Remember to get the ground wire stick well to the metal bar. Otherwise, it will block field of view of the camera).

9. Make sure the brain is dry enough so that the dye does not diffuse when touching the surface of the brain. Once the probe touch the surface of the target area, rezero LinLab.

11. Start to lower the probe with creeper function in LinLab, for at least 350 um (1um/s), so that all the channels are inside the tissue. Use Stimulus_toolkits and looming_toolkits to search for visual related neurons.

12. Once a good spot is found, remove or turn off the microscope and check the sound of spikes. When everything is ready, position the central monitor and turn off  micromanipulator to remove additional electrical noise.

13. The procedure of the recording without using fictrac: Passive viewing experiment is (1) start bonsai workflow (2) start recording on OpenEphys (3) start the stimulus with keypress **S**. In the case of spontanous experiment: (0) connect OpenEphys TTL channel 2 to Basler camera's TTL input (see closed loop experiment for details) (1) start recording on OpenEphys  (2)  start bonsai to activate the camera

14. get PFA, PBS and trypsin ready when the recording is finishing or when pulling out of the electrode back (no need to dilute PFA but trypsin is stored in 10x)

15. Once the recording is done, try to pull the electrode back (1um/s). If the brian is too dry, trying adding saline. If the probe was tangled by hemolymph too much, then added protease to remove it (but do add saline right afterward so that the brain tissue can be more or less intact).  

16. Add PBS into dissection stage and then move the head to the stage. Note: it is actually fine to just cut off the labula complex and save dissection time. Finally place the brain into the 4% PFA and store it at the cold room overnight.


# Closed loop experiment (with fictrac)

0. Open pylon view and load the feature file of basler camera. Note: fictrac has a preference over some series number so to make 9618 as default camera for fictrac, we need to unplug other cameras except 9675 (in the case of 9675, unplug every cameras). 

1. open comment prompt and move to C:\src\fictrac\bin\Release> to standby with fictrac.exe config_zball_camera1_9618_176x176_144Hz.txt

2. open bonsai workflow closed_loop_rotation_2_target.bonsai to standby

3. connect OpenEphys TTL channel 2 to Basler camera's TTL input (red cable of OpenEphys connecting to purple cable of Basler camera)

4. Start recording on OpenEphys GUI (the following sequence is opposite as recording session without using fictrac. This sequence is meant to ensure all the camera sync pulse is captured by openEphys. At the expanse of recording the transition of screen when Bonsai is turned on and off). Notably, even with this preparation, the number of sync pulse collected by OpenEphys is different from the data saved in fictrac. For example, in a roughly one-hour recording, 5111564 data points are saved by fictrac. 511604 data points saved by OpenEphys. The data loss is 0.007623 % The difference is negligable but it means the mapping between the two is not exactly 1 to 1

5. type config_zball_camera1_9618_176x176_144Hz.txt on comment prompt

6. start bonsai workflow closed_loop_rotation_2_target.bonsai and press S to let fictrac data flow in.

7. After the recording session, first turn off Bonsai workflow. Then turned off 

8. Stop recording on OpenEphys GUI

# Multicamera filmming

1. Use bonVision based visual paradigm: open the designed workflow and use arduino to trigger the cameras (and then turn on fictrac if this is a closed-loop experiment). Note: there is a difference feature file for camera 1557 for fictrac or for pose estimation. The camera config files are stored at C:\src\fictrac\pose_estimate\setup2025

2. Use Unity based visual paradigm: Open pylon viewer and turn on camera 1557 so that we can load feature into the camera (somehow fictrac automatically use this camera so we load a particular feature file:432x480_hw in this camera for fictrac) and then close Pylon Viewer.

2-1. turn on Bonsai workflow multicamera_hw, and use the same steps to start the arduino with **C**  and press **R** for recording. Then run fictrac and then run python_script fictrac\VR_array\socket_zmq_republisher.py. And then start OpenEphys recording >>> connect the barcode arduino >>> run Unity files (Here is a problem with which monitors Unity to target. Belows is the details).

Press **Esc** to stop the Unity file (this seems to take around 10 mins to save and compress the data; Unity is buggy on this pc so needs to use task manager to shut down this software). Then turn off fictrac and then turn off bonsai workflow.

>When running Unity programme on Ephys setup. After pressing the record button on OpenEphys and bonsai (to acquire video data), connect the sivler USB to activate the barcode arduino. Then use barcode sequence to align ephys (TTL channel 3) with video data


# no-Multicamera filmming

The fundamental difference between mult-camera filming or not is whether to use arduino to trigger the camera. Thus, if only using one camera to capture the behaviour or when synchronising camera shutter is not important, then use feature file that does not come with **_hw**. The rest of the step should be more or less the same.

## analysis methods

The analysis methods are organised into a json file, which is created from a python file. This json file tells the notebook and python script how to analysis the data (with some boolean options) and include meta information of the experiment that is not included during data collection phase. 

Therefore, each project (or each experiment) has its own json file. Below explains what those analysis methods are about.

# The following is used in raw2si.py and spike_curation.py  

    "experimenter": "chiyu",

    "probe_type": "H10_stacked", # currently only supports P2 and H10_stacked 
    
    "motion_corrector": "testing", # either choose a algorithm to correct drift/motion or using "testing" to go through each method 
    
    "sorter_name": "kilosort4",
    
    "load_previous_methods": false,
    
    "analyse_good_channels_only": true, # whether to remove bad and noisy channels from analysis

    "remove_dead_channels": true,

    "interpolate_noisy_channels": false,

    "plot_traces": false,
    
    "load_raw_traces": false,

    "tmin_tmax": [
        0.0,
        -1.0
    ],
    
    "skip_motion_correction": true,

    "load_existing_motion_info": false
    
    "save_prepocessed_file": false, # whether to compress and save dataset after it goes through bandpass filter, common median reference, remove bad channels, and slice into a portion
    
    "load_prepocessed_file": false,
    
    "save_sorting_file": true,
    
    "load_sorting_file": false,

# The following is used in spike_curation.py  
    
    "save_analyser_to_disc": true,
    
    "load_analyser_from_disc": false,
    
    "postprocess_with_unwhitening_recording": true,
    
    "export_to_phy": true,
    
    "overwrite_existing_phy": true,
    
    "load_curated_spikes": true,
    
    "export_report": false,

# The following is used in decode_spikes.py

    "load_raw_trial_info": false,

    "include_MUA": true, # whether to include MUA when using the script decode_spikes.py
    
    "event_of_interest": "stim_onset", # chooose an event to align spikes with
    
    "analyse_stim_evoked_activity": true, # if false, then entire data will be analysed. Work in progress
    
    "experiment_name": "coherence",

# The following is used in analyse animal's behaviour

    "overwrite_curated_dataset": true, whether to overwrite the existing pickle file or not, which is generated from fictrac. If True, then the programme will overwrite the pickle file

    "save_output": whether to save any output during data analysis. If True, then save any output
    
    "fictrac_posthoc_analysis": whether the fictrac data comes from online tracking or posthoc analysis. If True, then entering the analysis pipeline for posthoc analysis.
    
    "use_led_to_align_stimulus_timing": whether to use independent light source to track stimulus onset and offset. If True, then entering the analysis pipeline for that. If False, the programme will try to use timestamp coming from trials and from cameras to isolate the onset (this can only be achieved when posthoc analysis is not applied)
    
    "align_with_isi_onset": whether to use align the stimulus-evoked response based on stimulus onset or ISI onset. If True, then aligning the data based on ISI onset (this only works with Bonsai projects where ISI is logged)
    
    "mark_jump_as_nan": whether to use fft to identify jumping event in fictrac data and mark the event as nan

    "active_trials_only":  work in progress (to include walking trials)

    "filtering_method": what kind of filter to apply for tracking animals
    
    "plotting_tbt_overview": whether to plot a heat map of animal's velocity trial by trial and two histograms of animals travel distance and optomotor index.
    
    "plotting_trajectory": whether to plot the trajectory of the animals in the experiment.
    
    "plotting_event_related_trajectory": whether to plot a heat map of animal's trajectory after the stimulus onset
    
    "plotting_deceleration_accerleration": whether to plot the average deceleration and accerleration onset of the animals (this is still under construction)

    "plotting_position_dependant_fixation": whether to plot animal's angular velocity in response time. This is used in conflict experiment
    
    "plotting_optomotor_response": whether to plot the animal's mean optomotor index across stimulus type and the trajectory of yaw rotation vector trial by trial.
    
    "load_experiment_condition_from_database": this is used in the jypter notebook to see whether you want to access google-sheet-based database. If False, all animals stored in the server will be included.
    
    "select_animals_by_condition": this is used in the jypter notebook to extract specific animals for analysis. If True, you need to specify what condition in a dictionary. If False, all animals in the database (google sheet) will be included.
    
    "analysis_by_stimulus_type": this is used in the python script to see whether you want to plot the velocity heatmap based on stimlus type or based on time. If True, the heatmap will be stored based on stimulus type.

