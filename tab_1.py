import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px

from dash.dependencies import Input, Output
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
        )


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