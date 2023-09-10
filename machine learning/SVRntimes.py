import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR  
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Convert SMILES to Morgan fingerprint 
def smiles2morgan(smiles_list):
  fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=1024) for smiles in smiles_list]
  return fingerprints

# Data normalization
scaler = MinMaxScaler()

data = pd.read_csv('data_path') 

rmse_list = []
r2_list = []
#slect test times
for i in range(x):

  # Randomly split data into train and validation
  train, val_test = train_test_split(data, test_size=0.2, random_state=i)
  
  # Data processing and model training 
  X_train, y_train = train['smiles'], train['tg']
  X_val, y_val = val_test['smiles'], val_test['tg']

  # Convert SMILES to fingerprint and normalize
  X_train, X_val = smiles2morgan(X_train), smiles2morgan(X_val)
  X_train = scaler.fit_transform(X_train)
  X_val = scaler.transform(X_val)

  # SVR model training and prediction
  svr = SVR(C=10, epsilon=0.2)
  svr.fit(X_train, y_train)
  pred = svr.predict(X_val)
  
  # Evaluate model 
  rmse = np.sqrt(mean_squared_error(y_val, pred))
  r2 = np.abs(r2_score(y_val, pred))
  
  # Append to list
  rmse_list.append(rmse)
  r2_list.append(r2)

# Convert list to string
rmse_str = ' '.join([str(num) for num in rmse_list])
r2_str = ' '.join([str(num) for num in r2_list])

# Write to file
with open('dataset/svr/12svr_result.csv', 'w') as f:
  f.write(rmse_str + '\n')
  f.write(r2_str)

print('Done')
