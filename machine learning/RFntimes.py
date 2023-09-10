import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

# Load training data
train_data = pd.read_csv('train_data.csv')

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

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(train_fps, train_target, test_size=0.2, random_state=42)

# Create random forest regression model
rf_model = RandomForestRegressor(n_estimators=100)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Evaluate the model on the test data
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Generate fingerprints for the actual test data
test_data = pd.read_csv('test_data.csv')
test_smiles = test_data['smiles']
test_target = test_data['tg']

test_fps = get_morgan_fingerprint(test_smiles)

# Make predictions on the actual test data
predictions = []
for _ in range(1000):
    preds = rf_model.predict(test_fps)
    predictions.append(preds)

average_predictions = np.mean(predictions, axis=0)

# Evaluate the model on the actual test data
rmse = np.sqrt(mean_squared_error(test_target, average_predictions))
r2 = r2_score(test_target, average_predictions)

# Save the results
result_df = pd.DataFrame({'tg': test_target, 'tg_pred': average_predictions})
result_df.to_csv('results.csv', index=False)

with open('metrics.txt', 'w') as f:
    f.write(f'RMSE: {rmse:.2f}\n')
    f.write(f'R2 Score: {r2:.2f}\n')

print(f' R2: {r2:.2f}, RMSE: {rmse:.2f}')

result_df = pd.DataFrame({'RMSE': [rmse], 'R2': [r2]})
result_df.to_csv('rf_results.csv', mode='a', header=not os.path.exists('rf_results.csv'), index=False)
