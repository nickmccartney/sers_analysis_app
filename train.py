import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import dash
from dash.dependencies import Input, Output, State
import dash_table
import dash_bootstrap_components as dbc

import base64
import datetime
import io
import plotly.graph_objs as go
import plotly.express as px
import cufflinks as cf


from app import app
from app import server

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import sklearn
import scipy as sp
from pandas import MultiIndex
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

import pickle

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

def tabTrain():
    return html.Div([
        html.H3('Upload training data'),
        dcc.Graph(id='test-data'),
        dcc.Upload(
            id='test-upload',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),  

        html.Div(
            [
                dbc.Row(dbc.Col(html.H2('Pipeline Assembly'))),
                dbc.Row(
                    [
                        dbc.Col(html.H3('Scalers')),
                        dbc.Col(html.H3('Decomposers')),
                        dbc.Col(html.H3('Classifiers'))
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(
                            dbc.Checklist(
                                id='scalers-check',
                                options=[
                                    {'label': 'None', 'value': 'None'},
                                    {'label': 'Standard', 'value': 'stdScaler'},
                                    {'label': 'Min-Max', 'value': 'MinMaxScaler'},
                                    {'label': 'Max Absolute Value', 'value': 'MaxAbsScaler'}
                                ]
                            )
                        )),

                        dbc.Col(html.Div(
                            children = [
                            dbc.Checklist(
                                id='decomp-check',
                                options=[
                                    {'label': 'None', 'value': 'None'},
                                    {'label': 'Linear PCA', 'value': 'linearPCA'},
                                    {'label': 'Polynomial PCA', 'value': 'polyPCA'},
                                    {'label': 'Sigmoid PCA', 'value': 'sigmoidPCA'},
                                    {'label': 'Cosine PCA', 'value': 'cosinePCA'}
                                ]
                            ),
                            html.Div('Number of PCA Components:'),
                            dcc.Input(
                                id='PCA_n',
                                type='number',
                                value=3,
                                min = 1
                            )]
                        )),

                        dbc.Col(html.Div(
                            dbc.Checklist(
                                id='classif-check',
                                options=[
                                    {'label': 'None', 'value': 'None'},
                                    {'label': 'k-Nearest Neighbors', 'value': 'kNN'},
                                    {'label': 'Support Vector Classifier', 'value': 'SVC'}
                                ]
                            ),
                        )),
                    ]
                )
            ]
        ),

        html.Div(id='disp-choices'),  
        
    ],
    style={ 'margin-left': '150px',
            'margin-right': '150px'
        }
    )

def parse_data(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8'))).T
    df.columns = df.iloc[0]
    df = df.drop(['Raman Shift'])
    # sample_idx = [n for n in np.arange(len(df.index))]
    # midx = pd.MultiIndex.from_product([[molecule], [i], sample_idx], names=('Molecule', 'Concentration', 'Sample'))
    # df = df.set_index(midx)
    # data = pd.concat([data, df])
    return df


@app.callback(Output('test-data', 'children'),
              Input('test-upload', 'contents'),
              Input('test-upload', 'filename'))
def display_test_data(contents, filename):
    # return html.Div(value)
    test_DS = parse_data(contents, filename)
    return dcc.Graph(
                id='test-graph',
                figure=px.line(x=test_DS.columns.values, y=list(test_DS.values[0]))
            )

@app.callback(  Output('disp-choices', 'children'),
                Input('scalers-check', 'value'),
                Input('decomp-check', 'value'),
                Input('classif-check', 'value'),
                Input('PCA_n', 'value')
            )
def display_choices(scalerList, decomposerList, classifierList, n):
    from sklearn import preprocessing
    scalers = { 'stdScaler': preprocessing.StandardScaler(),
                'MinMaxScaler': preprocessing.MinMaxScaler(),
                'MaxAbsScaler': preprocessing.MaxAbsScaler()}

    from sklearn import decomposition
    
    decomposers = { 'linearPCA': decomposition.KernelPCA(kernel='linear', n_components = n),
                    'polyPCA': decomposition.KernelPCA(kernel='poly', n_components = n),
                    'rbfPCA': decomposition.KernelPCA(kernel='rbf', n_components = n),
                    'sigmoidPCA': decomposition.KernelPCA(kernel='sigmoid', n_components = n),
                    'cosinePCA': decomposition.KernelPCA(kernel='cosine', n_components = n)}

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    estimators = {  'kNN': KNeighborsClassifier(n_neighbors = 5),
                    'SVC': SVC()}

    # model = pipe.fit(X_values, y_labels)
    # pickle.dump(model, open('model.sav', 'wb'))
    # [str(scalers[i]) for i in scalerList]
    pipes = []

    for i in scalerList:
        for j in decomposerList:
            for k in classifierList:
                
                pipe = sklearn.pipeline.Pipeline([('scaler', scalers[i]), ('pca', decomposers[j]), ('est', estimators[k])])
                pipes = [*pipes, pipe]
    # pickle.dump(pipe, open('model.sav', 'wb'))
    return ','.join([str(p) for p in pipes])
                
                