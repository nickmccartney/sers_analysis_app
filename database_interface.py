import pandas as pd 
import sqlite3
from sqlite3 import Error

__all__ = [
    'create_connection',
    'list_datasets',
    'select_dataset',
    'store_dataset',
]

def create_connection():
    db_name = 'datasets.db'                     # FIXME: Fixed name, stored within cwd
    connection = None
    try:
        connection = sqlite3.connect(db_name)
        print("Connection to SERS DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection

def list_datasets():
    conn = create_connection()
    c = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [
        v[0] for v in c.fetchall()
        if v[0] != "sqlite_sequence"
    ]
    c.close()
    conn.close()
    return tables

def select_dataset(name):
    conn = create_connection()
    if name in list_datasets():
        dataset = pd.read_sql('SELECT * from {}'.format(name), conn)    # FIXME: security risk, cannot figure out how to use safer methods to fill in variable
        # restore midx
        midx = pd.MultiIndex.from_arrays([dataset['Molecule'], dataset['Concentration'], dataset['Sample']], names=('Molecule', 'Concentration', 'Sample'))  
        dataset = dataset.set_index(midx)
        dataset.drop(midx.names, axis=1, inplace=True)
        # restore float type of column name
        dataset.columns = [float(val) for val in dataset.columns]
    else:
        dataset = pd.DataFrame()                            # FIXME: is this an acceptable method to generate new dataset
    conn.close()
    return dataset

def store_dataset(dataset, name):
    conn = create_connection()
    dataset.to_sql(name, conn, if_exists='replace', index=True, index_label=dataset.index.names)
    conn.close()


# NOTE: random steps to force a functional dataset into database for testing
import numpy as np

name = 'Ocean_Insights'
dataset = select_dataset(name)
molecule = 'BPE : Methyl-Orange'
concentration = '1:1'

df = pd.read_csv('~/Desktop/spring_2021/ee_494/sers_data/AllSpectraData_2021-02-24_12-18-24_BPE_Methyl_Orange_1_1_wo_base.txt')
df = df.transpose()                     
df.columns = df.iloc[0]                 # use raman shift row as column index
df.drop('Raman Shift', inplace=True)    # raman shift row is unecessary 


if concentration in dataset.index.get_level_values(0).unique():                                 # check if concentration is already included in dataset
    n0 = int(dataset.loc[concentration].index[-1].replace('{} SP_'.format(molecule), '')) + 1   # find last value SP_n to start indexing sample at SP_n0 = n+1
else:
    n0 = 0                                                                                      # otherwise, sample_idx starts at SP_0

sample_idx = ['SP_{}'.format(n) for n in np.linspace(n0, n0+len(df.index), len(df.index), endpoint=False, dtype=int)]  # construct array of sample indicies
molecule_idx = [molecule]*len(sample_idx)
concentration_idx = [concentration]*len(sample_idx)                                             # construct array of concentration indices
midx = pd.MultiIndex.from_arrays([molecule_idx, concentration_idx, sample_idx], names=('Molecule', 'Concentration', 'Sample'))      # combine concentration & sample in multi index
df = df.set_index(midx)                 # assign multi index to df

dataset = pd.concat([dataset, df])
store_dataset(dataset, name)

if concentration in dataset.index.get_level_values(0).unique():                                 # check if concentration is already included in dataset
    n0 = int(dataset.loc[concentration].index[-1].replace('{} SP_'.format(molecule), '')) + 1   # find last value SP_n to start indexing sample at SP_n0 = n+1
else:
    n0 = 0                                                                                      # otherwise, sample_idx starts at SP_0

sample_idx = ['SP_{}'.format(n) for n in np.linspace(n0, n0+len(df.index), len(df.index), endpoint=False, dtype=int)]  # construct array of sample indicies
molecule_idx = [molecule]*len(sample_idx)
concentration_idx = [concentration]*len(sample_idx)                                             # construct array of concentration indices
midx = pd.MultiIndex.from_arrays([molecule_idx, concentration_idx, sample_idx], names=('Molecule', 'Concentration', 'Sample'))      # combine concentration & sample in multi index
df = df.set_index(midx)                 # assign multi index to df

dataset = pd.concat([dataset, df])
store_dataset(dataset, name)