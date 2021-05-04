import pandas as pd 
import numpy as np
import sqlite3
from sqlite3 import Error

__all__ = [
    'create_connection',
    'list_datasets',
    'select_dataset',
    'store_dataset',
    'list_models',
    'select_model',
    'store_model',
]

def create_connection(db_name):
    ''' 
        create connection to any local database storing user data
         - {str} db_name: filename of local sql database
    '''

    connection = None
    try:
        connection = sqlite3.connect(db_name)
        print("Connection to SERS DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection

###
#   Dataset Interfacing
###

def list_datasets():
    '''
        generate list of datasets stored in local sql database
    '''

    conn = create_connection('datasets.db')     # using db_name of 'datasets.db' since working with datasets
    c = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [
        v[0] for v in c.fetchall()
        if v[0] != "sqlite_sequence"
    ]
    c.close()
    conn.close()
    return tables

def select_dataset(name):
    '''
        select particular dataframe among datasets.db
         - {str} name: selection of dataframe to obtain
    '''

    conn = create_connection('datasets.db')     # using db_name of 'datasets.db' since working with datasets
    if name in list_datasets():
        dataset = pd.read_sql('SELECT * from {}'.format(name), conn)    # FIXME: security risk, cannot figure out how to use safer methods to fill in variable
        # restore midx
        midx = pd.MultiIndex.from_arrays([dataset['Molecule'], dataset['Concentration'], dataset['Sample']], names=('Molecule', 'Concentration', 'Sample'))  
        dataset = dataset.set_index(midx)
        dataset.drop(midx.names, axis=1, inplace=True)
        # restore float type of column name
        dataset.columns = [float(val) for val in dataset.columns]
    else:
        dataset = pd.DataFrame()                            
    conn.close()
    return dataset

def store_dataset(dataset, name):
    '''
        save/update dataset within database
         - {dataframe} dataset: df to store
         - {str} name: name to reference sql table for dataset 
    '''

    conn = create_connection('datasets.db')     # using db_name of 'datasets.db' since working with datasets
    dataset.to_sql(name, conn, if_exists='replace', index=True, index_label=dataset.index.names)
    conn.close()

###
# Model Interfacing
###

def list_models():
    '''
        generate list of models stored in local sql database
    '''

    conn = create_connection('models.db')     # using db_name of 'models.db' since working with models
    c = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [
        v[0] for v in c.fetchall()
        if v[0] != "sqlite_sequence"
    ]
    c.close()
    conn.close()
    return tables

def select_model(name):
    '''
        select particular model among models.db
         - {str} name: selection of model to obtain
    '''

    conn = create_connection('models.db')     # using db_name of 'models.db' since working with models
    if name in list_datasets():
        dataset = pd.read_sql('SELECT * from {}'.format(name), conn)    
    else:
        dataset = pd.DataFrame()                            
    conn.close()
    return dataset

def store_model(dataset, name):
    conn = create_connection('models.db')     # using db_name of 'models.db' since working with models
    dataset.to_sql(name, conn, if_exists='replace', index=True, index_label=dataset.index.names)
    conn.close()


###
# Simple Method for Data Export 
###

def export_dataset(name):
    '''
        save sql table containing dataset as '.xlsx' file
         - {str} name: name referencing particular dataset
    '''

    dataset = select_dataset(name)
    filtered_dataset = dataset
    for column in dataset.columns:
        if dataset[column].isnull().values.any():
            filtered_dataset.drop([column], inplace=True, axis=1)
    dataset = filtered_dataset.sort_index()
    molecules = dataset.index.get_level_values('Molecule').to_numpy()
    concentrations = dataset.index.get_level_values('Concentration').to_numpy()
    groups = np.unique(np.array([[molecule, concentration] for molecule, concentration in zip(molecules, concentrations)]), axis=0)
    with pd.ExcelWriter(f'{name}.xlsx') as writer:
        for group in groups:
            data = dataset.loc[(group[0],group[1])]
            print(str(group[0]), str(group[1]))
            data.to_excel(writer, sheet_name='{}_{}'.format(str(group[0]).replace(':','_'), str(group[1]).replace(':','_')))
