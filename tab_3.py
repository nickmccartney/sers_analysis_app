import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import dash
from dash.dependencies import Input, Output, State


from app import app
from app import server
import database_interface as dbi
import data_analysis as da

import pandas as pd
import numpy as np
import sklearn
import scipy as sp
from pandas import MultiIndex
import pickle

from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.svm import SVC


import plotly.graph_objects as go
import base64
import io
import dash_table

def render_tab():
    dataset_options = [{'label': name, 'value': name} for name in dbi.list_datasets()]

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H3("Dataset"),
                dcc.Dropdown(
                    id='test-dataset',
                    options=dataset_options,
                    placeholder='Select a Dataset...'
                ),
                html.Br(),

                html.H3("Testing Data"),
                dcc.Upload(
                    id='upload-test-data',
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
                    multiple=False  # Allow multiple files to be uploaded
                ),

                html.Br(),

                html.H3("Results"),
                html.Div(id='output-table'),


            ],
                style={
                    'background-color': 'lightgrey',
                    'padding': '25px'
                },
                width=4
            ),


            dbc.Col([
                html.Div(dcc.Graph(
                    id='test-data-graph',
                    figure={},
                ), id='test-data'
                ),
            ],
                style={
                    'background-color': 'lightsalmon',
                    'padding': '25px',
                    'height': '100%'
                },
                width=8,
            ),

        ])

    ],
        style={
            'background-color': 'default',
            'padding': '30px'
        }
    )


def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df



@app.callback(Output('test-data-graph', 'figure'),
              Input('upload-test-data', 'contents'),
              Input('upload-test-data', 'filename'))
def test_data_graph(contents, filename):
    if contents:
        contents = contents
        filename = filename
        df = parse_data(contents, filename)
        x_axis = df['Raman Shift']
        df.drop('Raman Shift', axis=1, inplace=True)

        fig = go.Figure()
        fig.update_layout(title_text='Testing Data Spectra')
        fig.update_xaxes(title_text='Raman Shift cm^-1')
        fig.update_yaxes(title_text='Relative Intensity')

        for signal in df.columns:
            y_axis = df[signal]
            fig.add_trace(go.Scatter(x=x_axis, y=y_axis, name=signal))

        return fig

    return {}


@app.callback(Output('output-table', 'children'),
              Input('test-dataset', 'value'),
              Input('upload-test-data', 'contents'),
              Input('upload-test-data', 'filename'))
def output_table(dataset_value, contents, filename):
    if(dataset_value and contents):

        model_frame = dbi.select_model(dataset_value)       # select training model
        df = parse_data(contents, filename)
        df.drop('Raman Shift', axis=1, inplace=True)      # drop raman shift
        df.dropna(axis='columns')                         # remove null values
        data = df.T

        # Molecule classification
        mol_model = pickle.loads(model_frame.iloc[0]['Pipeline'])  # iloc[0] b/c molecule is first in table
        mol_test = data.values
        mol_class = mol_model.predict(mol_test)




        X_class = pd.DataFrame()
        X_class['Trace'] = df.columns
        X_class['Molecule'] = mol_class
        X_class['Concentration'] = 'Concentration'



        # Concentration classification
        trace_names = data.columns          # for output table

        for i in range(len(df.columns)):
            mol_idx = model_frame.index[model_frame['Molecule'] == mol_class[i]].tolist()   # molecule columns index in model_frame
            conc_model = pickle.loads(model_frame.iloc[mol_idx[0]]['Pipeline'])             # pipeline for specific molecule
            conc_test = data.values
            conc_class = conc_model.predict(conc_test)
            X_class['Concentration'][i] = conc_class[i]
            # X_class = X_class.append({'Molecule': mol_class[i], 'Concentration': conc_class[i], 'Trace': trace_names[i]}, ignore_index=True)



        ### Create datatable dataframe to return

        # X_class = X_class.append({'Molecule': mol_class, 'Concentration': conc_class}, ignore_index=True)

        # X_class['Molecule'] = mol_model.predict(mol_test)
        # X_class['Concentration'] = 'Concentration'





        return(
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in X_class.columns],
                id='classification-table',
                data=X_class.to_dict('records'),
                style_cell={'textAlign': 'left'}
            ),
        )
