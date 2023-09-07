import pandas as pd 
from rdkit import Chem 
from rdkit.Chem import AllChem 

from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPRegressor 
from sklearn.metrics import mean_squared_error, r2_score 
import numpy as np  

from sklearn.preprocessing import MinMaxScaler # data normalization

# convert smiles strings to Morgan fingerprints
def smiles2morgan(smiles_list):
  fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=1024) for smiles in smiles_list]
  return fingerprints

# normalization 
scaler = MinMaxScaler() 

# read data
data = pd.read_csv('data_path') 

rmse, r2 = [], []

for i in range(n): # n-fold cross validation
  
  # split data 
  train, val_test = train_test_split(data, test_size=0.2, random_state=i)  

  X_train, y_train = train['smiles'], train['tg'] 
  X_val, y_val = val_test['smiles'], val_test['tg']
  
  X_train, X_val = smiles2morgan(X_train), smiles2morgan(X_val) # convert SMILES to fingerprints
  
  X_train = scaler.fit_transform(X_train) # normalize train data
  X_val = scaler.transform(X_val) 
  
  # model training and prediction
  mlp = MLPRegressor(max_iter= n, early_stopping=True)
  mlp.fit(X_train, y_train)

  pred = mlp.predict(X_val)
  
  # model evaluation
  rmse.append(np.sqrt(mean_squared_error(y_val, pred)))
  r2.append(np.abs(r2_score(y_val, pred)))

  # save results
  df = pd.DataFrame({'rmse': rmse, 'r2': r2})
  df.to_csv('data_path', index=False)
