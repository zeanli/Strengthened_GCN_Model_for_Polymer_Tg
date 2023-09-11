import numpy as np
import csv
import pandas as pd
import random
from rdkit import Chem


def randomize_smile(sml,max_len=100):
    """Function that randomizes a SMILES sequnce. This was adapted from the
    implemetation of E. Bjerrum 2017, SMILES Enumeration as Data Augmentation
    for Neural Network Modeling of Molecules.
    Args:
        sml: SMILES sequnce to randomize.
    Return:
        randomized SMILES sequnce or
        nan if SMILES is not interpretable.
    """
    try:
        m = Chem.MolFromSmiles(sml)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m, ans)
        smiles = Chem.MolToSmiles(nm, canonical=False)
        i = 0
        while len(smiles)>max_len:
            m = Chem.MolFromSmiles(sml)
            ans = list(range(m.GetNumAtoms()))
            np.random.shuffle(ans)
            nm = Chem.RenumberAtoms(m, ans)
            smiles = Chem.MolToSmiles(nm, canonical=False)
            i = i+1
            if i>5:
                break
        if len(smiles)>max_len:
            return sml
        else:
            return smiles
    except:
        return np.nan
        
        
def canonical_smile(sml):
#"""Helper Function that returns the RDKit canonical SMILES for a input SMILES sequnce.
#Args:
#    sml: SMILES sequence.
#Returns:
#    canonical SMILES sequnce."""
    a = Chem.MolFromSmiles(sml)
    return Chem.MolToSmiles(a, canonical = True)
    

    
df_train = pd.DataFrame()
df_test = pd.DataFrame()
#data_path need tobe replaced with the dataset you want to train
with open('data_path','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    column_0 = [row[0] for row in csvReader]
    df_train['A']=(column_0)
    
with open('data_path','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    column_1 = [row[1] for row in csvReader]
    df_train['B']=(column_1)
    
with open('data_path','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    column_2 = [row[0] for row in csvReader]
    df_test['A']=(column_2)

with open('data_path','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    column_3 = [row[1] for row in csvReader]
    df_test['B']=(column_3)

'''
with open('../data/yanshi.csv','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    column_0 = [row[0] for row in csvReader]
    df_train['A']=(column_0)
    
with open('../data/yanshi.csv','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    column_1 = [row[1] for row in csvReader]
    df_train['B']=(column_1)
'''
    
#----personalized strengthen,choose
df_train_all = pd.DataFrame()
n_times = 1  #n-1



#----canonical smiles
df_train_1_copytmp = df_train.copy(deep = True)
df_train_1_copytmp['A'] = df_train_1_copytmp['A'].map(lambda x: canonical_smile(x))
df_train_all = pd.concat([df_train_all, df_train_1_copytmp], ignore_index = True)

#----randomized
for i in range(0, n_times):
    df_train_1_copytem = df_train.copy(deep = True)
    df_train_1_copytem['A'] = df_train_1_copytem['A'].map(lambda x: randomize_smile(x))
    df_train_all = pd.concat([df_train_all, df_train_1_copytem], ignore_index = True)
print('df_train_all:',df_train_all)

df_train_all = pd.concat([df_train_all, df_test], ignore_index = True)

df_train_all.to_csv('.csv'.format(n_times), header =False, index = False)
    
    
'''smlstr = np.asarray(smlstr)
df = pd.DataFrame(columns=('canonical','1'))

for m in range(1,10):
    for i in range(len(smlstr)):
        tmp = smlstr[i]
        canonical_smi = canonical_smile(tmp)
        new_smiles = randomize_smile(tmp)
        df.loc[i,['canonical','1']] = [canonical_smi, new_smiles]
        df.to_csv('savr_path)'''
        
        
        
        
        
