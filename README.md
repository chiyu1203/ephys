### ephys
General anaconda environment for all setups and data analyses

Install the latest anaconda version for your operating system (https://www.anaconda.com/products/individual).

Open the anaconda prompt and create a virtual environment via conda
````
conda create --name ephys --channel conda-forge python=3.10
conda activate ephys
conda update -n base -c defaults conda
conda config --add channels conda-forge
conda config --set channel_priority strict
````
Then installing spikeinterface and ibllib from source. When cloning their repositories, I usually store them in a GitHub folder in the Document.
If you want to follow this structure, then the following command will be useful. For further info, check out their github page or documentation webpage.

Note1: ideally, we only need to install ibllib spikeinterface via pip. However, since kilosort4 is released, many bugs appeared in spikeinterface (as well as in kilosort4), by installing from source, we can get updated version sooner (the downside is that we need to double check whether the newer version introducing additional bugs by ourselves).

Note2: I forked ibllib because I wanted to test some ploting functions that is only for our own lab so we do not clone the original ibl repo.

Note3: ibllib use scipy 1.12 but installing one of these packages **open-ephys-python-tools zarr docker cuda-python numcodecs hdbscan** 
needs scipy 1.13 so I hopes there is no conflict between them 
````
cd Documents\GitHub
git clone https://github.com/SpikeInterface/spikeinterface.git
cd spikeinterface
pip install -e .
cd ..
pip install git+https://github.com/NeuralEnsemble/python-neo.git
pip install git+https://github.com/SpikeInterface/probeinterface.git
pip install git+https://github.com/chiyu1203/ibllib.git
````

Then installing other dependencies.
````
pip install open-ephys-python-tools zarr docker cuda-python numcodecs hdbscan
````
If you have a good GPU and wants to install kilosort. Here is the instruction.
````
python -m pip install kilosort[gui]
pip uninstall torch
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
````

Probably dont need the packages below anymore
````
conda install --yes -c conda-forge -v ipython jupyter pytest ipympl 
````
## data collection

# Preparation

1. Use dehumilifier to reduce the humidity overnight and heat up the room in the morning.

2. prepare locusts saline, dyes, PBS, PFA, tin foil, and a beaker with trypsin for electrode, 2 pipettes (one for loading dye and the other one for loading saline) and 1 pipetteman. Place these reagents in a box with ice.

3. turn on IR LED,


# Multicamera filmming

1. Use bonVision based visual paradigm: open the designed workflow and use arduino to trigger the cameras (and then turn on fictrac if this is a closed-loop experiment). Note: there is a difference feature file for camera 1557 for fictrac or for pose estimation.

2. Use Unity based visual paradigm: Open pylon viewer and turn on camera 1557 so that we can load feature into the camera (somehow fictrac automatically use this camera so we load a particular feature file:432x480_hw in this camera for fictrac) and then close Pylon Viewer.

2-1. turn on Bonsai workflow multicamera_hw, and use the same steps to start the arduino and press **R** for recording. Then run fictrac and then run python_script fictrac\VR_array\socket_zmq_republisher.py. And then start OpenEphys recording >>> connect the barcode arduino >>> run Unity files (Here is a problem with which monitors Unity to target. Belows is the details). 

Press **Esc** to stop the Unity file (this seems to take around 10 mins to save and compress the data; Unity is buggy on this pc so needs to use task manager to shut down this software). Then turn off fictrac and then turn off bonsai workflow.

In addition, I shall not think about running unity and ephys before I add this feature 

>when running Unity programme on Ephys setup. One big problem is that stimuli are presented when the monitors are closed up so ~~either I editted the unity programme to make the control scence to present at the second monitors and VR scene at the main monitors~~ not easy. I shall completely remove the control scence and directly load parameters in the Swarm scene. A quick way to shut down the second monitor is just to unplug it.
>how to implement stimulus alignment with Ephys without using sync pulse is not clear.


# no-Multicamera filmming

The fundamental difference between mult-camera filming or not is whether to use arduino to trigger the camera. Thus, if only using one camera to capture the behaviour or when synchronising camera shutter is not important, then use feature file that does not come with **_hw**. The rest of the step should be more or less the same.

