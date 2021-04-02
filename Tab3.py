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

import pickle

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

def Test():
    return html.Div([
        html.H3('Upload testing data'),
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