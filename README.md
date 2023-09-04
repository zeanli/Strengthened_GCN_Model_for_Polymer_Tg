# strengthened-GCN-model-for-polymer-Tg-prediction-
This project contains graph con model which can be used to predict Tg of polymers.
This project contains graph con model which can be used to predict Tg of polymers. This project references the MPNN model from the open source project chemprop (https://github.com/chemprop/chemprop), as well as the CMC_GCN model from the paper Predicting Critical Micelle Concentrations for Surfactants Using Graph Convolutional Neural Networks（https://doi.org/10.1021/acs.jpcb.1c05264）.
<h2>Requirements</h2>
To use CMC_GCN model,you will need:
python＞=3.7,CUDA＞=10.2
<h2>Install</h2>

<h2>RUN</h2>
If you need to train the model, run workflow_test.py, where you need to change the data path to the corresponding file path. To augment the data, run strengthen_ntimes.py, noting to change the dataset path and augmentation multiples. To generate psmiles data, run strengthen_psmiles.py. To perform personalized differential augmentation, run strengthen_personalized.py.
