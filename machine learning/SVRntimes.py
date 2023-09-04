import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# SMILES转分子指纹
def smiles2morgan(smiles_list):
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=1024) for smiles in
                    smiles_list]
    return fingerprints


# 分子指纹归一化
scaler = MinMaxScaler()

# 读取数据
data = pd.read_csv('dataset/12times.csv')
rmse_list = []
r2_list = []

for i in range(10):
     # 随机划分数据
    train, val_test = train_test_split(data, test_size=0.2, random_state=i)

    # 数据处理和模型训练预测
    X_train, y_train = train['smiles'], train['tg']
    X_val, y_val = val_test['smiles'], val_test['tg']

    # SMILES转指纹、归一化
    X_train, X_val = smiles2morgan(X_train), smiles2morgan(X_val)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

        # SVR模型训练和预测
    svr = SVR(C=10, epsilon=0.2)
    svr.fit(X_train, y_train)
    pred = svr.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = np.abs(r2_score(y_val, pred))

    # append到列表
    rmse_list.append(rmse)
    r2_list.append(r2)

# 将列表转换成字符串
rmse_str = ' '.join([str(num) for num in rmse_list])
r2_str = ' '.join([str(num) for num in r2_list])

# 写入文件
with open('dataset/svr/12svr_result.csv', 'w') as f:
    f.write(rmse_str + '\n')
    f.write(r2_str)

print('Done')