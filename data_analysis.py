import pandas as pd
import numpy as np
from sklearn import pipeline
import pickle
from sklearn.pipeline import Pipeline

__all__ = [
    'append_to_dataset',
    'serialize_model',
]

def append_to_dataset(dataset, df, molecule, concentration):
    '''
        properly format and then append new sample data to dataset
         - {dataframe} dataset: dataset which new data is appended to
         - {dataframe} df: dataframe containing new sample data
         - {str} molecule: label for molecule type
         - {str} concentration: label for concentration type
    '''

    df = df.transpose()      
    df.columns = df.iloc[0]                 # use raman shift row as column index

    df.drop('Raman Shift', inplace=True)    
    
    if concentration in dataset.index.get_level_values(0).unique():                                 # check if concentration is already included in dataset
        n0 = int(dataset.loc[concentration].index[-1].replace('{} SP_'.format(molecule), '')) + 1   # find last value SP_n to start indexing sample at SP_n0 = n+1
    else:
        n0 = 0                                                                                      # otherwise, sample_idx starts at SP_0

    sample_idx = ['SP_{}'.format(n) for n in np.linspace(n0, n0+len(df.index), len(df.index), endpoint=False, dtype=int)]  # construct array of sample indicies
    molecule_idx = [molecule]*len(sample_idx)
    concentration_idx = [concentration]*len(sample_idx)                                             # construct array of concentration indices
    midx = pd.MultiIndex.from_arrays([molecule_idx, concentration_idx, sample_idx], names=('Molecule', 'Concentration', 'Sample'))      # combine concentration & sample in multi index
    df = df.set_index(midx)                 # assign multi index to df

    dataset = pd.concat([dataset, df])      # concatenate df onto dataset

    return dataset


### Pipeline dicts
from sklearn import preprocessing
scalers = { 'None': preprocessing.FunctionTransformer(),
            'stdScaler': preprocessing.StandardScaler(),
            'MinMaxScaler': preprocessing.MinMaxScaler(),
            'MaxAbsScaler': preprocessing.MaxAbsScaler()}

from sklearn import decomposition
n=3
decomposers = { 'None': preprocessing.FunctionTransformer(),
                'linearPCA': decomposition.KernelPCA(kernel='linear', n_components = n),
                'polyPCA': decomposition.KernelPCA(kernel='poly', n_components = n),
                'rbfPCA': decomposition.KernelPCA(kernel='rbf', n_components = n),
                'sigmoidPCA': decomposition.KernelPCA(kernel='sigmoid', n_components = n),
                'cosinePCA': decomposition.KernelPCA(kernel='cosine', n_components = n)}

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
estimators = {  'None': preprocessing.FunctionTransformer(),
                'kNN': KNeighborsClassifier(n_neighbors = 5),
                'SVC': SVC()}

def serialize_model(df, scaler_value, decomposer_value, estimator_value):   # Serialize trained models
    '''
        returns a fitted pipeline in a serialized format
        - {dataframe} df: Dataframe, first index level serves as labels for fitted model
        - {str} scaler_value: key for scalers dict
        - {str} decomposer_value: key for decomposer dict
        - {str} estimator_value: key for estimators dict 
    '''

    pipe = Pipeline([('scaler', scalers[scaler_value]), ('pca', decomposers[decomposer_value]), ('estimator', estimators[estimator_value])])    # Assembles a pipeline
    fitted = pipe.fit(df.values, df.index.get_level_values(0))                                                  # fits model to concentration features of current mol.
    serialized = pickle.dumps(fitted)                                                                           # Serializes fitted model  
    return serialized