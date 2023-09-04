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
    
smlstr = []
#test_idx = [0,1,2,3,4]
#test_idx = [272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 298]
test_idx = np.random.randint(0,201,18)
train_idx = []
all_idx = []
#test_idx = [5, 23, 40, 66, 81, 95, 111, 119, 132, 137, 148, 154, 156, 157, 160, 176, 182, 190]


with open('cmc_dataset_202.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        smlstr.append(row[0])
        
#----set index for train and test        
dataset_size = len(smlstr) 

#test_idx = np.random.randint(0,dataset_size,dataset_size//11)
print('test_idx:',test_idx)

all_idx = np.arange(dataset_size)
train_idx = np.asarray([x for x in all_idx if x not in test_idx])
print('train_idx:',train_idx)
    
df_0 = pd.DataFrame()

with open('cmc_dataset_202.csv','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    column_0 = [row[0] for row in csvReader]
    df_0['A']=(column_0)
    
with open('cmc_dataset_202.csv','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    column_1 = [row[1] for row in csvReader]
    df_0['B']=(column_1)
    
with open('cmc_dataset_202.csv','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    column_2 = [row[2] for row in csvReader]
    df_0['C']=(column_2)

with open('cmc_dataset_202.csv','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    column_3 = [row[3] for row in csvReader]
    df_0['D']=(column_3)
df_0['D'] = df_0['D'].apply(int)
print(df_0)

#----create df for train and test
df_test = pd.DataFrame()
df_train = pd.DataFrame()

for i in test_idx:
    df_tem = df_0.loc[[i]]
    df_test = df_test.append(df_tem, ignore_index = True)
    
df_test.drop(columns = ['C','D'],inplace = True)
#print(df_test)

####print randomized test set
#df_test.to_csv('../data/600/saliency/olefins_esters_test.csv',sep=',',index=False,header=False)


for i in train_idx:
    df_tem2 = df_0.loc[[i]]
    df_train = df_train.append(df_tem2, ignore_index = True)
print('df_train:',df_train)    
    
#----personalized strengthen
df_train_all = pd.DataFrame()
first_times = 4  #3-1
second_times = 8  #7-1
third_times = 12  #12-1

first_node = 12
second_node = 20

#----first times
df_train['D'] = df_train['D'].apply(int)
df_train_1 = df_train.loc[lambda x: x['D'] < first_node]
print('df_train_1:',df_train_1)
#smlstr = np.array(df_train_1['A'])
#print('smlstr:', smlstr)

#----canonical smiles
df_train_1_copytmp = df_train_1.copy(deep = True)
df_train_1_copytmp['A'] = df_train_1_copytmp['A'].map(lambda x: canonical_smile(x))
df_train_all = pd.concat([df_train_all, df_train_1_copytmp], ignore_index = True)

#----randomized
for i in range(0, first_times):
    df_train_1_copytem = df_train_1.copy(deep = True)
    df_train_1_copytem['A'] = df_train_1_copytem['A'].map(lambda x: randomize_smile(x))
    df_train_all = pd.concat([df_train_all, df_train_1_copytem], ignore_index = True)
print('df_train_all:',df_train_all)
  
#----second times
df_train_2 = df_train.loc[lambda x: (x['D'] >= first_node) & (x['D'] < second_node)]
print('df_train_2:',df_train_2)

df_train_2_copytmp = df_train_2.copy(deep = True)
df_train_2_copytmp['A'] = df_train_2_copytmp['A'].map(lambda x: canonical_smile(x))
df_train_all = pd.concat([df_train_all, df_train_2_copytmp], ignore_index = True)

for i in range(0, second_times):
    df_train_2_copytem = df_train_2.copy(deep = True)
    df_train_2_copytem['A'] = df_train_2_copytem['A'].map(lambda x: randomize_smile(x))
    df_train_all = pd.concat([df_train_all, df_train_2_copytem], ignore_index = True)
print('df_train_all:',df_train_all)


#----third times
df_train_3 = df_train.loc[lambda x: x['D'] >= second_node]
print('df_train_3:',df_train_3)

df_train_3_copytmp = df_train_3.copy(deep = True)
df_train_3_copytmp['A'] = df_train_3_copytmp['A'].map(lambda x: canonical_smile(x))
df_train_all = pd.concat([df_train_all, df_train_3_copytmp], ignore_index = True)

for i in range(0, third_times):
    df_train_3_copytem = df_train_3.copy(deep = True)
    df_train_3_copytem['A'] = df_train_3_copytem['A'].map(lambda x: randomize_smile(x))
    df_train_all = pd.concat([df_train_all, df_train_3_copytem], ignore_index = True)
print('df_train_all:',df_train_all)
    
#    df_train_1 = pd.concat([df_train_1_copytem, df_train_1], ignore_index = True)
#print('df_train_1',df_train_1)

df_train_all.drop(columns = ['C','D'],inplace = True)
df_train_all = pd.concat([df_train_all, df_test], ignore_index = True)
df_train_all.to_csv('../data/cmc/202_personal.csv', header =False, index = False)

    
'''smlstr = np.asarray(smlstr)
df = pd.DataFrame(columns=('canonical','1'))

for m in range(1,10):
    for i in range(len(smlstr)):
        tmp = smlstr[i]
        canonical_smi = canonical_smile(tmp)
        new_smiles = randomize_smile(tmp)
        df.loc[i,['canonical','1']] = [canonical_smi, new_smiles]
        df.to_csv('../data/300/data_strength_300_3_{}stren.csv'.format(m))'''
        
        
        
        
        
