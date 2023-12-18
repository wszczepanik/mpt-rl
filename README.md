# Installation
Tested on Ubuntu 22.04 on WSL 2.

## Requirements
Install requirements
```bash
sudo apt install g++ python3 cmake ninja-build git
sudo apt install libboost-all-dev libprotobuf-dev protobuf-compiler pybind11-dev
```

## Download ns3
Install ns3
```bash
wget https://www.nsnam.org/releases/ns-allinone-3.40.tar.bz2
tar xfj ns-allinone-3.40.tar.bz2
cd ns-allinone-3.40/ns-3.40
```

## ns3-ai
Clone modified ns3-ai repository
```
git clone https://github.com/wszczepanik/ns3-ai.git contrib/ai
```

## Build ns3
Building might take few minutes
```
./ns3 configure
./ns3 build
```
Errors due to not using all modules used by ns3-ai can be ignored.

## Python
(Optional) Create venv with Python version matching system one (ns3-ai requirement). Example below requires conda.
```bash
conda create -n mpt python=3.10
conda activate mpt
```
Install ns3-ai modules
```
pip install -e contrib/ai/python_utils
pip install -e contrib/ai/model/gym-interface/py
```

## mpt-rl
Download repository
```
git clone https://github.com/wszczepanik/mpt-rl.git scratch/reinforcement-learning
```
Install Python libraries from `requirements.txt`
```
cd scratch/rllib-integration
pip install -r requirements.txt
```
