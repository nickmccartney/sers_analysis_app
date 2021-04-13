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


# dataset = select_dataset('TEST')
# print(dataset.index)