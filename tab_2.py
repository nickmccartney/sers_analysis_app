import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import dash
from dash.dependencies import Input, Output, State
import dash_table
from dash.exceptions import PreventUpdate

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
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(                                   # Explanation Div
                                    [
                                        html.H3('Model Training'),
                                        html.Hr(),
                                        html.P('This section is for training machine learning models on existing datasets. To begin, first select a dataset from the dropdown menu below.'),
                                        html.P('Each model defaults to a certain set of scalers, decomposers, and estimators. You may view any changes to a model in the visualiser by selecting the respective model. If you are not familiar with these items, you may leave them as their default values, or read the documentation here:'),
                                        dcc.Link(href='https://scikit-learn.org/stable/index.html#')
                                    ],
                                    style={
                                        'background-color': 'lightgrey',
                                        'padding': '25px'
                                    },
                                ),
                                html.Div(                                   # Dataset Select Div
                                    [                                  
                                        html.Div([
                                            html.H3('1. Select Dataset'),
                                            dcc.Dropdown(
                                                id='dataset-training-select',
                                                options=dataset_options,
                                                placeholder='Select a Dataset...'
                                            ),
                                            html.Div('Select a dataset to begin'),
                                        ]),
                                    ],
                                    style={
                                        'background-color': '#b3ffb3',
                                        'padding': '25px'
                                    },
                                ),
                                html.Div(                                   # Model Select Div
                                    [                                  
                                        html.H3('2. Verify Models'),
                                        dash_table.DataTable(
                                            style_cell={
                                                'whiteSpace': 'normal',
                                                'height': 'auto',
                                            },
                                            columns=[
                                                {'id': 'Classification Task', 'name': 'Task', 'editable': False},
                                                {'id': 'Molecule', 'name': 'Molecule', 'editable': False},
                                                {'id': 'Scaler', 'name': 'Scaler', 'presentation': 'dropdown'},
                                                {'id': 'Decomposer', 'name': 'Decomposer', 'presentation': 'dropdown'},
                                                {'id': 'Estimator', 'name': 'Estimator', 'presentation': 'dropdown'},
                                            ],
                                            id='pipeline-table',
                                            editable=True,
                                            row_selectable='single',
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
                                            data=[{ 'Classification Task': '---', 
                                                    'Molecule': '---', 
                                                    'Scaler': '---',
                                                    'Decomposer': '---',
                                                    'Estimator': '---'},
                                                { 'Classification Task': '---', 
                                                    'Molecule': '---', 
                                                    'Scaler': '---',
                                                    'Decomposer': '---',
                                                    'Estimator': '---'},
                                                { 'Classification Task': '---', 
                                                    'Molecule': '---', 
                                                    'Scaler': '---',
                                                    'Decomposer': '---',
                                                    'Estimator': '---'}]
                                        ),
                                        html.Div('Select a model from the table to view in the data visualiser. Click the below button to save your assembled models to the database.'),
                                        dbc.Button(
                                            'Train All Models', 
                                            id='train-button',
                                        ),
                                        html.Div(id='editing-prune-data-output'),
                                        html.Div(id='disp-train-success'),
                                    ],
                                    style={
                                        'background-color': 'lightblue',
                                        'padding': '25px'
                                    },
                                ),
                            ]
                        ),
                        width = 5
                    ),
                    dbc.Col(                                        # PCA Section
                        html.Div(
                            [
                                html.H3('Data Visualiser'),
                                html.Div(id = 'disp-dataset'),      # PCA Plot
                                dcc.Graph(
                                    id='pca-plot',
                                    style={'width': 'auto', 'height': '70vh'}
                                ),
                            ],
                            style={
                                'background-color': 'LightSalmon',
                                'padding': '25px',
                                # 'height': '100%'
                            },
                        ),
                    ),
                ],
            ),

        ],
        
        style={
            'background-color': 'default',
            'padding': '30px'
        }
    ),



###### Callback to plot PCA using defined pipe ######
@app.callback(  Output('pca-plot', 'figure'),             
                Input('dataset-training-select', 'value'),
                Input('pipeline-table', 'data'),
                Input('pipeline-table', 'selected_rows'),
)
def display_dataset(dataset_value, pipe_df, row):
    if not dataset_value:
        raise PreventUpdate
    if not pipe_df:
        raise PreventUpdate
    
    # scaler_value, decomposer_value, task_value, mol
    if row:
        row = row[0]
    else:
        raise PreventUpdate
    
    scaler_value = pipe_df[row]['Scaler']
    decomposer_value = pipe_df[row]['Decomposer']
    task_value = pipe_df[row]['Classification Task']
    mol = pipe_df[row]['Molecule']
    
    ### 
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
        px.scatter_3d(
            pca_components, x=0, y=1, z=2, color=df.index.get_level_values(0).values,
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3', 'color': 'Concentration'}
        )
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


    
###### Callback to update datatable with listed pipelines ######
@app.callback(  Output('pipeline-table', 'data'),             
                Input('dataset-training-select', 'value'),
)
def disp_table(dataset_value):
    if not dataset_value:
        raise PreventUpdate
    
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
        model_frame.to_dict('records')
    )



###### Callback to construct & store models to database on button click ######
@app.callback(  Output('disp-train-success', 'children'),
                Input('train-button', 'n_clicks'),             
                State('dataset-training-select', 'value'),
                State('pipeline-table', 'data'),
)
def assemble_models(clicked, dataset_value, table_data):
    # if not clicked:
    #     return 'Please select a model'
    if not dataset_value:
        raise PreventUpdate
       
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