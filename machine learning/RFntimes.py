import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import numpy as np

# Load training data
train_data = pd.read_csv('data_path')

# Extract SMILES strings and target values from training data
train_smiles = train_data['smiles']
train_target = train_data['tg']


# Generate Morgan fingerprints for training data
def get_morgan_fingerprint(smiles_list, radius=3):
    mols = []
    fps = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius)
                fps.append(fp)
            else:
                print(f"Invalid SMILES: {smiles}")
        except Exception as e:
            print(f"Error processing SMILES: {smiles}")
            print(f"Error message: {str(e)}")
    return fps

# Generate Morgan fingerprints for training data
train_fps = get_morgan_fingerprint(train_smiles)

# Create random forest regression model
rf_model = RandomForestRegressor(n_estimators=500)

# cross validation
y_pred = cross_val_predict(rf_model, train_fps, train_target, cv=5)

cv_rmse = np.sqrt(mean_squared_error(train_target, y_pred))
cv_r2 = r2_score(train_target, y_pred)

print("交叉验证的平均RMSE:", cv_rmse)
print("交叉验证的平均R2得分:", cv_r2)

# 训练模型
rf_model.fit(train_fps, train_target)
print("交叉验证的平均RMSE:", cv_rmse)
print("交叉验证的平均R2得分:", cv_r2)


# Generate fingerprints for test data
test_data = pd.read_csv('data_path')
X_test = test_data['smiles']

test_data = pd.read_csv('tests27.csv')


test_smiles = test_data['smiles']
test_target = test_data['tg']


test_fps = []
test_mols = []
for smiles in test_data['smiles']:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            test_mols.append(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3)
            test_fps.append(fp)
        else:
            print(f"Invalid SMILES: {smiles}")
    except Exception as e:
        print(f"Error processing SMILES: {smiles}")
        print(f"Error message: {str(e)}")

# Make predictions on test data
predictions = []
for _ in range(1000):
    preds = rf_model.predict(test_fps)
    predictions.append(preds)

average_predictions = sum(predictions) / len(predictions)

# Evaluate model on test data
rmse = mean_squared_error(test_target, average_predictions, squared=False)
r2 = r2_score(test_target, average_predictions)

# save
result_df = pd.DataFrame({'tg': test_target, 'tg_pred': average_predictions})
result_df.to_csv('data_path', index=False)

with open('metrics.txt', 'w') as f:
    f.write(f'RMSE: {rmse:.2f}\n')
    f.write(f'R2 Score: {r2:.2f}\n')
print(f' R2: {r2:.2f},RMSE:{rmse:.2f}')
result_df = pd.DataFrame({'RMSE': [rmse], 'R2': [r2]})
result_df.to_csv('data_path', mode='a', header=not os.path.exists('rf_results.csv'), index=False)