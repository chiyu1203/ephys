# ephys
General anaconda environment for all setups and data analyses

Install the latest anaconda version for your operating system (https://www.anaconda.com/products/individual).

Open the anaconda prompt and create a virtual environment via conda
````
conda create --name tracking_analysis --channel conda-forge python=3.10
conda activate tracking_analysis
conda update -n base -c defaults conda
conda config --add channels conda-forge
conda config --set channel_priority strict
python -m pip install kilosort[gui]
pip uninstall torch
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
pip install open-ephys-python-tools spikeinterface zarr docker cuda-python numcodecs hdbscan
````
Probably dont need the command below anymore
conda install --yes -c conda-forge -v ipython jupyter matplotlib tqdm pandas pytest fooof neurodsp ipympl 