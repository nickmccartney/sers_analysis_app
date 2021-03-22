import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

import plotly.express as px
import pandas as pd
import base64
import io

from app import app
from app import server
import database_interface as dbi

def render_tab():
    dataset_options = [{'label': name, 'value': name} for name in dbi.list_datasets()]
    dataset_options.append({'label': 'Create New (enter name below...)', 'value': 'NEW'})

    return html.Div([
        html.Label("Dataset"),
        dcc.Dropdown(
            id='select-dataset',
            options=dataset_options,
            placeholder='Select a Dataset...'
        ),
        html.Div([
            dcc.Input(
                id='new-dataset',
                type='text',
                placeholder='Enter a dataset name...',
            )
        ],
            id='new-dataset-container',
            style={'display': 'none'}
        ),

        html.Br(),

        # html.Label("Molecule"),
        html.Div([
            # html.Label("Molecule"),
            dcc.Dropdown(
                id='select-molecule',
                placeholder='Select a Molecule...'
            )
        ],
            id='select-molecule-container',
            style={'display': 'block'}
        ),

        html.Div([
            dcc.Input(
                id='new-molecule', 
                type='text', 
                placeholder='Enter a molecule...'
            )
        ],
            id='new-molecule-container',
            style={'display': 'none'}
        ),

        html.Br(),

        # html.Label("Concentration"),
        html.Div([
            dcc.Dropdown(
                id='select-concentration',
                placeholder='Select a Concentration...'
        )
        ],
            id='select-concentration-container',
            style={'display': 'block'}
        ),

        html.Div(
            dcc.Input(
                id='new-concentration', 
                type='text', 
                placeholder='Enter a concentration...'
            ),
            id='new-concentration-container',
            style={'display': 'none'}
        ),

        dcc.Upload(
            id='upload-data',
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
            multiple=False # Allow multiple files to be uploaded
        ),

        html.Button(
            'Import',
            id='submit-import',
            n_clicks=0
        ),


        html.H1(id='intermediate-value')

        # html.Div(id='dataset-spectrum'),
    ])

@app.callback(Output('new-dataset-container', 'style'),
              Input('select-dataset', 'value'))
def update_dataset_input(value):
    enabled_style = {'display': 'block'}
    disabled_style = {'display': 'none'}

    if value == 'NEW':
        return enabled_style
    else:
        return disabled_style


@app.callback(Output('select-molecule', 'options'),
              Output('select-molecule-container', 'style'),
              Output('new-molecule-container','style'),
              Input('select-dataset', 'value'),
              Input('select-molecule', 'value'))
def update_molecule_input(dataset_value, molecule_value):
    enabled_style = {'display': 'block'}
    disabled_style = {'display': 'none'}
    options = []

    if dataset_value == 'NEW':
        return options, disabled_style, enabled_style
    elif dataset_value != None:
        dataset = dbi.select_dataset(dataset_value)

        molecules = dataset.index.get_level_values('Molecule').unique().to_numpy()              # list unique molecule labels within dataset
        options = [{'label': name, 'value': name} for name in molecules]
        options.append({'label': 'Create New (enter name below...)', 'value': 'NEW'})  # allow choice to assign new molecule label

        if molecule_value == 'NEW':
            return options, enabled_style, enabled_style
        elif molecule_value != None:
            return options, enabled_style, disabled_style
        else:
            return options, enabled_style, disabled_style
    else:
        return options, disabled_style, disabled_style
      
    
@app.callback(Output('select-concentration', 'options'),
              Output('select-concentration-container', 'style'),
              Output('new-concentration-container', 'style'),
              Input('select-dataset', 'value'),
              Input('select-molecule', 'value'),
              Input('select-concentration', 'value'))
def update_concentration_input(dataset_value, molecule_value, concentration_value):
    enabled_style = {'display': 'block'}
    disabled_style = {'display': 'none'}
    options = []


    if dataset_value == 'NEW':
        return options, disabled_style, enabled_style
    elif dataset_value != None and molecule_value != None:
        if molecule_value == 'NEW':
            return options, disabled_style, enabled_style
        else:
            dataset = dbi.select_dataset(dataset_value)

            molecules = dataset.loc[molecule_value].index.get_level_values('Concentration').unique().to_numpy()              # list unique molecule labels within dataset
            options = [{'label': name, 'value': name} for name in molecules]
            options.append({'label': 'Create New (enter name below...)', 'value': 'NEW'})  # allow choice to assign new molecule label

            if concentration_value == 'NEW':
                return options, enabled_style, enabled_style
            elif concentration_value != None:
                return options, enabled_style, disabled_style
            else:
                return options, enabled_style, disabled_style
    else:
        return options, disabled_style, disabled_style


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
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), delimiter = r'\s+')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df


@app.callback(Output('intermediate-value', 'children'),
              Input('submit-import', 'n_clicks'),
              State('select-dataset', 'value'),
              State('new-dataset', 'value'),
              State('select-molecule', 'value'),
              State('new-molecule', 'value'),
              State('select-concentration', 'value'),
              State('new-concentration', 'value'),
              State('upload-data', 'contents'),
              State('upload-data', 'filename'))
def import_data(n_clicks, select_dataset, new_dataset, select_molecule, new_molecule, select_concentration, new_concentration, contents, filename):

    df = pd.DataFrame()
    dataset_label = None
    molecule_label = None
    concentration_label = None

    if select_dataset == 'NEW':
        dataset_label = new_dataset
    else:
        dataset_label = select_dataset

    if select_molecule == 'NEW':
        molecule_label = new_molecule
    else:
        molecule_label = select_molecule

    if select_concentration == 'NEW':
        concentration_label = new_concentration
    else:
        concentration_label = select_concentration


    if contents:
        contents = contents
        filename = filename
        df = parse_data(contents, filename)


    if dataset_label!=None and molecule_label!=None and concentration_label!=None and not df.empty:
        ret = 'Import Success'
    else:
        ret = 'No Import Yet'

    return ret