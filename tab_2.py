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
import pickle


def render_tab():
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
            dbc.Row(html.Div(id='disp-pipeline-table')),
            dbc.Row(dbc.Button('Train Models',
                                id='train-button',)),
            dbc.Row(html.Div(id='editing-prune-data-output')),
            dbc.Row(html.Div(id='disp-train-success')),
        ],
        
        style={
            'background-color': 'default',
            'padding': '30px'
        }
    ),



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
    pca_fit = pca_pipe.fit(X)
    
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


    
###### Callback to update datatable ######
@app.callback(  Output('disp-pipeline-table', 'children'),             
                Input('dataset-training-select', 'value'),
)
def disp_table(dataset_value):
    dataset = dbi.select_dataset(dataset_value)

    model_frame = pd.DataFrame( data = [['Molecule', 'All', 'MinMaxScaler', 'cosinePCA', 'SVC']], 
                                columns=['Classification Task', 'Molecule', 'Scaler', 'Decomposer', 'Estimator']) ### Assembles dataframe for models
    for i in dataset.index.unique(0).values:
        model_frame = model_frame.append({  'Classification Task': 'Concentration', 
                                            'Molecule': i, 
                                            'Scaler': 'MinMaxScaler',
                                            'Decomposer': 'cosinePCA',
                                            'Estimator': 'SVC'}, ignore_index = True)    # Appends new row to array
        
    return (
        dash_table.DataTable(
            columns=[
                {'id': 'Classification Task', 'name': 'Task', 'editable': False},
                {'id': 'Molecule', 'name': 'Molecule', 'editable': False},
                {'id': 'Scaler', 'name': 'Scaler', 'presentation': 'dropdown'},
                {'id': 'Decomposer', 'name': 'Decomposer', 'presentation': 'dropdown'},
                {'id': 'Estimator', 'name': 'Estimator', 'presentation': 'dropdown'},
            ],
            id='pipeline-table',
            data=model_frame.to_dict('records'),
            editable=True,
            dropdown={
                'Scaler': {
                    'options': [
                        {'label': 'None', 'value': 'None'},
                        {'label': 'Standard', 'value': 'stdScaler'},
                        {'label': 'Min-Max', 'value': 'MinMaxScaler'},
                        {'label': 'Max Absolute Value', 'value': 'MaxAbsScaler'}
                    ]
                },
                'Decomposer': {
                    'options': [
                        {'label': 'None', 'value': 'None'},
                        {'label': 'Linear PCA', 'value': 'linearPCA'},
                        {'label': 'Polynomial PCA', 'value': 'polyPCA'},
                        {'label': 'Sigmoid PCA', 'value': 'sigmoidPCA'},
                        {'label': 'Cosine PCA', 'value': 'cosinePCA'}
                    ]
                },
                'Estimator': {
                    'options': [
                        {'label': 'None', 'value': 'None'},
                        {'label': 'k-Nearest Neighbors', 'value': 'kNN'},
                        {'label': 'Support Vector Classifier', 'value': 'SVC'}
                    ]
                },
            },
            css=[{"selector": ".Select-menu-outer", "rule": "display: block !important"}],
        )
    )



###### Callback to construct & store models to database ######
@app.callback(  Output('disp-train-success', 'children'),
                Input('train-button', 'n_clicks'),             
                State('dataset-training-select', 'value'),
                State('pipeline-table', 'data'),
)
def assemble_models(clicked, dataset_value, table_data):
    dataset = dbi.select_dataset(dataset_value)
    dataset = dataset.dropna(axis='columns')                          # Remove null values
    
    df = pd.DataFrame.from_dict(table_data)
    
    arr=[]
    model_frame = df.drop(columns=['Scaler', 'Decomposer', 'Estimator'])
    for i in df.index.values:
        if df.iloc[i]['Classification Task'] != 'Molecule':
            data = dataset.loc[df.iloc[i]['Molecule']]      # Gets dataframe for conc. task
        else:
            data = dataset
        
        model = da.serialize_model(data, df.iloc[i]['Scaler'], df.iloc[i]['Decomposer'], df.iloc[i]['Estimator'])

        arr = [*arr, model]
    model_frame['Pipeline'] = arr
    
    dbi.store_model(model_frame, dataset_value)
    return "Saved " + dataset_value
    # return dash_table.DataTable(
    #     id='some-table',
    #     columns=[{"name": str(i), "id": str(i)} for i in model_frame.columns],
    #     data=model_frame.to_dict('records'),
    # )


# import pprint
# @app.callback(Output('editing-prune-data-output', 'children'),    ### Prints raw data
#               Input('pipeline-table', 'data'))
# def display_output(rows):
#     pruned_rows = []
#     for row in rows:
#         # require that all elements in a row are specified
#         # the pruning behavior that you need may be different than this
#         if all([cell != '' for cell in row.values()]):
#             pruned_rows.append(row)

#     return html.Div([
#         html.Div('Raw Data'),
#         html.Pre(pprint.pformat(rows)),
#         html.Hr(),
#         html.Div('Pruned Data'),
#         html.Pre(pprint.pformat(pruned_rows)),
#     ])