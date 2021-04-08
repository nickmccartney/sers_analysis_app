import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import dash
from dash.dependencies import Input, Output, State
import dash_table

from app import app
from app import server
import database_interface as dbi
import data_analysis as da

import pandas as pd
import numpy as np 
import sklearn
import scipy as sp
from pandas import MultiIndex

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.svm import SVC


def Train():
    dataset_options = [{'label': name, 'value': name} for name in dbi.list_datasets()]
    
    return html.Div(
        [
            dbc.Row(dbc.Col(html.Label("Dataset"))),
            dbc.Row(dbc.Col(
                dcc.Dropdown(
                id='select-dataset-training',
                options=dataset_options,
                placeholder='Select a Dataset...')
            )),
            # 'div-disp-dataset'
            
            dbc.Row(dbc.Col(
                html.Div(
                    id = 'div-disp-dataset'
                )
            )),
        
            # dbc.Row(dbc.Col(html.H2('Pipeline Assembly'))),
            # dbc.Row(
            #     [
            #         dbc.Col(html.H3('Scalers')),
            #         dbc.Col(html.H3('Decomposers')),
            #         dbc.Col(html.H3('Classifiers'))
            #     ]
            # ),
            # dbc.Row(
            #     [
            #         dbc.Col(html.Div(
            #             dbc.Checklist(
            #                 id='scalers-check',
            #                 options=[
            #                     {'label': 'None', 'value': 'None'},
            #                     {'label': 'Standard', 'value': 'stdScaler'},
            #                     {'label': 'Min-Max', 'value': 'MinMaxScaler'},
            #                     {'label': 'Max Absolute Value', 'value': 'MaxAbsScaler'}
            #                 ]
            #             )
            #         )),

            #         dbc.Col(html.Div(
            #             children = [
            #             dbc.Checklist(
            #                 id='decomp-check',
            #                 options=[
            #                     {'label': 'None', 'value': 'None'},
            #                     {'label': 'Linear PCA', 'value': 'linearPCA'},
            #                     {'label': 'Polynomial PCA', 'value': 'polyPCA'},
            #                     {'label': 'Sigmoid PCA', 'value': 'sigmoidPCA'},
            #                     {'label': 'Cosine PCA', 'value': 'cosinePCA'}
            #                 ]
            #             ),
            #             html.Div('Number of PCA Components:'),
            #             dcc.Input(
            #                 id='PCA_n',
            #                 type='number',
            #                 value=3,
            #                 min = 1
            #             )]
            #         )),

            #         dbc.Col(html.Div(
            #             dbc.Checklist(
            #                 id='classif-check',
            #                 options=[
            #                     {'label': 'None', 'value': 'None'},
            #                     {'label': 'k-Nearest Neighbors', 'value': 'kNN'},
            #                     {'label': 'Support Vector Classifier', 'value': 'SVC'}
            #                 ]
            #             ),
            #         )),
            #     ]
            # ),
            # dbc.Row(dbc.Col(html.Div(id='disp-choices'))),
        ],
        
        style={ 'margin-left': '150px',
            'margin-right': '150px'
        }
    ),
    
# @app.callback(  Output('disp-choices', 'children'),
#                 Input('scalers-check', 'value'),
#                 Input('decomp-check', 'value'),
#                 Input('classif-check', 'value'),
#                 Input('PCA_n', 'value')
#             )
# def display_choices(scalerList, decomposerList, classifierList, n):
#     from sklearn import preprocessing
#     scalers = { 'stdScaler': preprocessing.StandardScaler(),
#                 'MinMaxScaler': preprocessing.MinMaxScaler(),
#                 'MaxAbsScaler': preprocessing.MaxAbsScaler()}

#     from sklearn import decomposition
    
#     decomposers = { 'linearPCA': decomposition.KernelPCA(kernel='linear', n_components = n),
#                     'polyPCA': decomposition.KernelPCA(kernel='poly', n_components = n),
#                     'rbfPCA': decomposition.KernelPCA(kernel='rbf', n_components = n),
#                     'sigmoidPCA': decomposition.KernelPCA(kernel='sigmoid', n_components = n),
#                     'cosinePCA': decomposition.KernelPCA(kernel='cosine', n_components = n)}

#     from sklearn.neighbors import KNeighborsClassifier
#     from sklearn.svm import SVC
#     estimators = {  'kNN': KNeighborsClassifier(n_neighbors = 5),
#                     'SVC': SVC()}

#     # model = pipe.fit(X_values, y_labels)
#     # pickle.dump(model, open('model.sav', 'wb'))
#     # [str(scalers[i]) for i in scalerList]
#     pipes = []

#     for i in scalerList:
#         for j in decomposerList:
#             for k in classifierList:
                
#                 pipe = sklearn.pipeline.Pipeline([('scaler', scalers[i]), ('pca', decomposers[j]), ('est', estimators[k])])
#                 pipes = [*pipes, pipe]
#     # pickle.dump(pipe, open('model.sav', 'wb'))
#     return html.Div([
#         html.Div("You are using: " + ','.join([str(p) for p in pipes])),
#         dbc.Button(
#             "Confirm"
#         )
#     ]
#     )

@app.callback(Output('div-disp-dataset', 'children'),
             Input('select-dataset-training', 'value'))
def disp_dataset(dataset_value):
    test_DS = dbi.select_dataset(dataset_value)
    df = test_DS.loc['Fentanyl : Heroin']
    
    # pipe = Pipeline([('scaler', preprocessing.MinMaxScaler()), ('pca', decomposition.KernelPCA(kernel='cosine', n_components = 3)), ('est', SVC())])
    # pipe.fit(X_train, y_train)        # keep for actual ML
    
    # return dcc.Graph(
    #             id='pca-plot',
    #             figure=px.line(x=test_DS.columns.values, y=list(test_DS.values[0]))
    #         ) 

    df = df.dropna(axis='columns')   # FIXME: Resolve issue of different raman shifts

    X = df.iloc[:, 1:len(df.columns)]
    
    pca_pipe = Pipeline([('scaler', preprocessing.MinMaxScaler()), ('pca', decomposition.KernelPCA(kernel='cosine', n_components = 3))])
    pca_components = pca_pipe.fit_transform(X)
    
    return (
        html.Label("PCA plot using " + str(pca_pipe)),
        dcc.Graph(
                id='pca-plot',
                figure=
                    px.scatter_3d(
                    pca_components, x=0, y=1, z=2, color=df.index.get_level_values(0).values,
                    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3', 'color': 'Concentration'}
                ),
                style={'height': '700px'}
            ),
        )