import dash_html_components as html
import dash_core_components as dcc
import dash_daq as daq
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
            dbc.Row(
                [
                    dbc.Col(html.Div(       ### Left hand PCA plot
                            id = 'disp-dataset',
                        ),
                        width = 9
                    ),
                    
                    dbc.Col(                ### ... and a pipe in the right hand (Pipeline assembly)
                        html.Div([                          
                            html.Div([
                                html.H3('Dataset'),
                                dcc.Dropdown(
                                    id='dataset-training-select',
                                    options=dataset_options,
                                    placeholder='Select a Dataset...'
                                ),
                                dbc.RadioItems(
                                    id='task-radio',
                                    options=[
                                        {'label': 'Molecule', 'value': 'Molecule'},
                                        {'label': 'Concentration', 'value': 'Concentration'},
                                    ],
                                    value='Molecule',
                                ),
                                dbc.Collapse(
                                    [
                                    dcc.Dropdown(
                                        id='mol-select',
                                        options=[],       ### populate options in callback from dataset select
                                        placeholder='Select a Dataset...'
                                    ),
                                    ],
                                    id='mol-collapse'
                                ),
                                dbc.Button(
                                    "Advanced Options",
                                    id="pipe-collapse-button",
                                    # className="mb-3",
                                    # color="primary",
                                ),
                                dbc.Collapse(
                                    [
                                    html.Hr(),
                                    
                                    html.Div([
                                        html.H3('Scalers'),
                                        dbc.RadioItems(
                                            id='scalers-radio',
                                            options=[
                                                {'label': 'None', 'value': 'None'},
                                                {'label': 'Standard', 'value': 'stdScaler'},
                                                {'label': 'Min-Max', 'value': 'MinMaxScaler'},
                                                {'label': 'Max Absolute Value', 'value': 'MaxAbsScaler'}
                                            ],
                                            value='MinMaxScaler',       ### Defaults currently for concentration
                                        )
                                    ]),
                                    html.Hr(),
                                    
                                    html.Div([
                                        html.H3('Decomposers'),
                                        dbc.RadioItems(
                                            id='decomp-radio',
                                            options=[
                                                {'label': 'None', 'value': 'None'},
                                                {'label': 'Linear PCA', 'value': 'linearPCA'},
                                                {'label': 'Polynomial PCA', 'value': 'polyPCA'},
                                                {'label': 'Sigmoid PCA', 'value': 'sigmoidPCA'},
                                                {'label': 'Cosine PCA', 'value': 'cosinePCA'}
                                            ],
                                            value='cosinePCA',       ### Defaults currently for concentration
                                        ),
                                        html.Div('# PCA Components:'),
                                        dcc.Input(
                                            id='PCA_n',
                                            type='number',
                                            value=3,
                                            min = 1
                                        )
                                    ]),
                                    html.Hr(),

                                    html.Div([
                                        html.H3('Classifiers'),
                                        dbc.RadioItems(
                                            id='classif-radio',
                                            options=[
                                                {'label': 'None', 'value': 'None'},
                                                {'label': 'k-Nearest Neighbors', 'value': 'kNN'},
                                                {'label': 'Support Vector Classifier', 'value': 'SVC'}
                                            ],
                                            value='SVC',       ### Defaults currently for concentration
                                        ),
                                    ]),
                                    ],
                                    id="pipe-collapse",
                                ),
                                html.Div(id='disp-pipe-select')
                            ]),
                        ],
                        style={
                            'background-color': 'lightgrey',
                            'padding': '25px'
                        },
                        ),
                        # width = 'auto'
                    ),
                ]
            ),
            dbc.Row(dbc.Col(html.Div(id='disp-choices'))),
        ],
        
        style={
            'background-color': 'default',
            'padding': '30px'
        }
    ),
    
###### Callback to assemble all models ######
@app.callback(  Output('disp-choices', 'children'),   
                Input('scalers-radio', 'value'),
                Input('decomp-radio', 'value'),
                Input('classif-radio', 'value'),
                Input('PCA_n', 'value')
            )
def display_choices(scaler_value, decomposer_value, classifier_value, n):
    from sklearn import preprocessing
    scalers = { 'None': preprocessing.FunctionTransformer(),
                'stdScaler': preprocessing.StandardScaler(),
                'MinMaxScaler': preprocessing.MinMaxScaler(),
                'MaxAbsScaler': preprocessing.MaxAbsScaler()}

    from sklearn import decomposition
    
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

    # model = pipe.fit(X_values, y_labels)
    # pickle.dump(model, open('model.sav', 'wb'))
    # [str(scalers[i]) for i in scaler_value]
    pipes = []

    for i in scaler_value:
        for j in decomposer_value:
            for k in classifier_value:
                
                pipe = sklearn.pipeline.Pipeline([('scaler', scalers[i]), ('pca', decomposers[j]), ('est', estimators[k])])
                pipes = [*pipes, pipe]
    # pickle.dump(pipe, open('model.sav', 'wb'))
    return html.Div([
        html.Div("You are using: " + ','.join([str(p) for p in pipes])),
        dbc.Button(
            "Confirm"
        )
    ]
    )



###### Callback to plot PCA using defined pipe ######
@app.callback(  Output('disp-dataset', 'children'),             
                Input('dataset-training-select', 'value'),
                Input('scalers-radio', 'value'),
                Input('decomp-radio', 'value'),
                Input('task-radio', 'value'),
                Input('mol-select', 'value'),
)
def display_dataset(dataset_value, scaler_value, decomposer_value, task_value, mol):
    from sklearn import preprocessing
    scalers = { 'None': preprocessing.FunctionTransformer(),
                'stdScaler': preprocessing.StandardScaler(),
                'MinMaxScaler': preprocessing.MinMaxScaler(),
                'MaxAbsScaler': preprocessing.MaxAbsScaler()}

    from sklearn import decomposition
    n=3     ### Hard defined for 3D PCA plot
    decomposers = { 'None': preprocessing.FunctionTransformer(),
                    'linearPCA': decomposition.KernelPCA(kernel='linear', n_components = n),
                    'polyPCA': decomposition.KernelPCA(kernel='poly', n_components = n),
                    'rbfPCA': decomposition.KernelPCA(kernel='rbf', n_components = n),
                    'sigmoidPCA': decomposition.KernelPCA(kernel='sigmoid', n_components = n),
                    'cosinePCA': decomposition.KernelPCA(kernel='cosine', n_components = n)}

    ### Above dicts of pipe modifiers
    
    test_DS = dbi.select_dataset(dataset_value)
    if task_value == 'Molecule':
        df = test_DS
    else: 
        df = test_DS.loc[mol]
        
    X = df.dropna(axis='columns')   # FIXME: Resolve issue of different raman shifts
    
    pca_pipe = Pipeline([('scaler', scalers[scaler_value]), ('pca', decomposers[decomposer_value])])
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



###### Callback to collapse advanced settings panel ######
@app.callback(                          
    Output("pipe-collapse", "is_open"),
    [Input("pipe-collapse-button", "n_clicks")],
    [State("pipe-collapse", "is_open")],
)
def toggle_pipe_collapse(n, is_open):
    if n:
        return not is_open
    return is_open



###### Callback to uncollapse molecule selector when doing concentration task ######
@app.callback(                          
    Output("mol-collapse", "is_open"),
    [Input("task-radio", "value")],
    [State("mol-collapse", "is_open")],
)
def toggle_mol_collapse(task, is_open):
    if task == 'Concentration':
        return 1
    return 0



###### Callback to populate molecule selection dropdown ######
@app.callback(Output('mol-select', 'options'),      
              Input('dataset-training-select', 'value'),
              Input('mol-select', 'value')
)
def update_molecule_input(dataset_value, molecule_value):
    # enabled_style = {'display': 'block'}
    # disabled_style = {'display': 'none'}
    options = []

    if dataset_value != None:
        dataset = dbi.select_dataset(dataset_value)

        molecules = dataset.index.get_level_values('Molecule').unique().to_numpy()              # list unique molecule labels within dataset
        options = [{'label': name, 'value': name} for name in molecules]
        
        return options#, enabled_style, disabled_style
    else:
        return options#, disabled_style, disabled_style