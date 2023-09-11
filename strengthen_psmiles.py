import numpy as np
import csv
import pandas as pd
import random
from rdkit import Chem
from psmiles import PolymerSmiles as PS


def canonical_psmiles(sml):
#"""Helper Function that returns the RDKit canonical SMILES for a input SMILES sequnce.
#Args:
#    sml: SMILES sequence.
#Returns:
#    canonical SMILES sequnce."""
    ps = PS(sml)
    return ps


def dimer0_smiles(sml):
    
    ps = PS(sml)
    dimer_0 = ps.dimer(0)
    return dimer_0
    
def dimer1_smiles(sml):
    
    ps = PS(sml)
    dimer_1 = ps.dimer(1)
    return dimer_1    

def randomize_psmiles(sml):

    ps = PS(sml)
    random_ps = ps.randomize
    return random_ps



#----create df for train and test
df_test = pd.DataFrame()
df_train = pd.DataFrame()    
df_0 = pd.DataFrame()

test_idx = [271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295]
#test_idx = np.random.randint(0,9,2)
train_idx = []
all_idx = []
smlstr = []


with open('psmiles.csv','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    for row in csvReader:
        smlstr.append(row[0])

with open('psmiels.csv','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    column_0 = [row[0] for row in csvReader]
    df_0['A']=(column_0)
    
with open('psmiles.csv','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    column_1 = [row[1] for row in csvReader]
    df_0['B']=(column_1)
    
with open('psmiles.csv','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    column_2 = [row[2] for row in csvReader]
    df_0['C']=(column_2)

with open('psmiles.csv','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    column_3 = [row[3] for row in csvReader]
    df_0['D']=(column_3)
df_0['D'] = df_0['D'].apply(int)
print(df_0)

#----set index for train and test        
dataset_size = len(smlstr) 
print('test_idx:',test_idx)

all_idx = np.arange(dataset_size)
train_idx = np.asarray([x for x in all_idx if x not in test_idx])
print('train_idx:',train_idx)


#----generate test and train dataset
for i in test_idx:
    df_tem = df_0.loc[[i]]
    df_test = df_test.append(df_tem, ignore_index = True)
    
df_test.drop(columns = ['C','D'],inplace = True)

for i in train_idx:
    df_tem2 = df_0.loc[[i]]
    df_train = df_train.append(df_tem2, ignore_index = True)
print('df_train:',df_train) 


#----personalized strengthen
df_train_all = pd.DataFrame()
first_times = 2  #4-2
second_times = 6  #8-2
third_times = 10  #12-2

first_node = 12
second_node = 20

#smlstr = np.array(df_train_1['A'])
#print('smlstr:', smlstr)


#----canonical smiles
df_train_1_copytmp = df_train.copy(deep = True)
df_train_1_copytmp['A'] = df_train_1_copytmp['A'].map(lambda x: canonical_psmiles(x))
df_train_all = pd.concat([df_train_all, df_train_1_copytmp], ignore_index = True)

#----dimer0_1
df_train_2_copytem = df_train.copy(deep = True)
df_train_2_copytem['A'] = df_train_2_copytem['A'].map(lambda x: dimer0_smiles(x))
df_train_all = pd.concat([df_train_all, df_train_2_copytem], ignore_index = True)

df_train_3_copytem = df_train.copy(deep = True)
df_train_3_copytem['A'] = df_train_3_copytem['A'].map(lambda x: dimer1_smiles(x))
df_train_all = pd.concat([df_train_all, df_train_3_copytem], ignore_index = True)

#----first_times

df_train['D'] = df_train['D'].apply(int)
df_train_1 = df_train.loc[lambda x: x['D'] < first_node]


for i in range(0, first_times):
    df_train_11_copytem = df_train_1.copy(deep = True)
    df_train_11_copytem['A'] = df_train_11_copytem['A'].map(lambda x: randomize_psmiles(x))
    df_train_all = pd.concat([df_train_all, df_train_11_copytem], ignore_index = True)
print('df_train_all:',df_train_all)

#----second_times

df_train['D'] = df_train['D'].apply(int)
df_train_2 = df_train.loc[lambda x: (x['D'] >= first_node) & (x['D'] < second_node)]

for i in range(0, second_times):
    df_train_21_copytem = df_train_1.copy(deep = True)
    df_train_21_copytem['A'] = df_train_21_copytem['A'].map(lambda x: randomize_psmiles(x))
    df_train_all = pd.concat([df_train_all, df_train_21_copytem], ignore_index = True)
#print('df_train_all:',df_train_all)

#----third times
df_train_3 = df_train.loc[lambda x: x['D'] >= second_node]


for i in range(0, third_times):
    df_train_31_copytem = df_train_3.copy(deep = True)
    df_train_31_copytem['A'] = df_train_31_copytem['A'].map(lambda x: randomize_psmiles(x))
    df_train_all = pd.concat([df_train_all, df_train_31_copytem], ignore_index = True)
print('df_train_all:',df_train_all)

df_train_all.drop(columns = ['C','D'],inplace = True)
df_train_all = pd.concat([df_train_all, df_test], ignore_index = True)
df_train_all.to_csv('save_data_path', header =False, index = False)


'''
with open('../data/ps_test.csv','r') as  csvDataFile:
    csvReader = csv.reader(csvDataFile)        
    column_1 = [row[1] for row in csvReader]
    df_train['B']=(column_1)

    
#----personalized strengthen
df_train_all = pd.DataFrame()


#----canonical smiles
df_train_1_copytmp = df_train.copy(deep = True)
df_train_1_copytmp['A'] = df_train_1_copytmp['A'].map(lambda x: canonicalize(x))
df_train_all = pd.concat([df_train_all, df_train_1_copytmp], ignore_index = True)

#----randomized

df_train_1_copytem = df_train.copy(deep = True)
df_train_1_copytem['A'] = df_train_1_copytem['A'].map(lambda x: dimer(x,0))
df_train_all = pd.concat([df_train_all, df_train_1_copytem], ignore_index = True)
print('df_train_all:',df_train_all)

df_train_all = pd.concat([df_train_all, df_test], ignore_index = True)

df_train_all.to_csv('../data/ps_strengthen.csv'.format(n_times), header =False, index = False)
'''