# ephys
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
