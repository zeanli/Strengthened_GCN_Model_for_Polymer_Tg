import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# SMILES转Morgan分子指纹
def smiles2morgan(smiles_list):
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=1024) for smiles in
                    smiles_list]
    return fingerprints


# 分子指纹归一化
scaler = MinMaxScaler()

# 读取数据
data = pd.read_csv('data_path')

rmse, r2 = [], []
for i in range(10):
    # 划分数据
    train, val_test = train_test_split(data, test_size=0.2, random_state=i)

    X_train, y_train = train['smiles'], train['tg']
    X_val, y_val = val_test['smiles'], val_test['tg']


    X_train, X_val = smiles2morgan(X_train), smiles2morgan(X_val)


    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 模型训练和预测
    mlp = MLPRegressor(max_iter=10000, early_stopping=True)
    mlp.fit(X_train, y_train)

    pred = mlp.predict(X_val)

    # 模型评估
    rmse.append(np.sqrt(mean_squared_error(y_val, pred)))
    r2.append(np.abs(r2_score(y_val, pred)))

# 保存结果
df = pd.DataFrame({'rmse': rmse, 'r2': r2})
df.to_csv('data_path', index=False)
