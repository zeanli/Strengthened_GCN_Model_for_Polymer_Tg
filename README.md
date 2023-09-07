<h1>Strengthened GCN Model for Polymer Tg Prediction</h1>                                          
This project contains graph con model which can be used to predict Tg of polymers.
This project references the MPNN model from the open source project chemprop (https://github.com/chemprop/chemprop), as well as the CMC_GCN model from the paper Predicting Critical Micelle Concentrations for Surfactants Using Graph Convolutional Neural Networks（https://doi.org/10.1021/acs.jpcb.1c05264）.
<h2>Requirements</h2>
To use CMC_GCN model,you will need:
python＞=3.7,CUDA＞=10.2
<h2>Install</h2>
Set up conda environment and clone the github repo

## Create a new environment
```
$ conda create --name n python=3.7
$ conda activate n
```
## Install requirements
```
$ pip install pytorch=1.9.1=py3.7_cuda10.2_cudnn7.6.5_0
$ pip install dgllife=0.2.6
$ pip install scikit-learn=1.0.2
$ conda install -c conda-forge rdkit=2020.09.1.0
$ conda install -c conda-forge tensorboard
```
## Clone the source code of Strengthened GCN
```
git clone https://github.com/zeanli/strengthened-GCN-model-for-polymer-Tg-prediction-
```
## Usage
To train the model, run  
```
$ python workflow_test.py
```
To augment the data, run
```
python strengthen_ntimes.py
```
To generate psmiles data, run
```
python strengthen_psmiles.py
```
To perform personalized differential augmentation, run
```
pythobn strengthen_personalized.py
```
