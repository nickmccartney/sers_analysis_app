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
        html.Br(),

        html.Div(id='label-entry'),

        # html.Div(id='dataset-spectrum'),
    ])

@app.callback(Output('label-entry', 'children'),
              Input('select-dataset', 'value'))
def render_molecule_entry(value):
    if value == 'NEW':
        return html.Div([
            dcc.Input(id='new-dataset', type='text', placeholder=''),
            html.Br(),
            html.Label("Molecule"),
            dcc.Input(id='new-molecule', type='text', placeholder=''),
            html.Br(),
            html.Label("Concentration"),
            dcc.Input(id='new-concentration', type='text', placeholder='')
        ])
        
    elif value != None:
        global dataset                                                                          # FIXME: should find better solution : https://dash.plotly.com/sharing-data-between-callbacks
        dataset = dbi.select_dataset(value)

        molecules = dataset.index.get_level_values('Molecule').unique().to_numpy()              # list unique molecule labels within dataset
        molecule_options = [{'label': name, 'value': name} for name in molecules]
        molecule_options.append({'label': 'Create New (enter name below...)', 'value': 'NEW'})  # allow choice to assign new molecule label

        return html.Div([
            html.Label("Molecule"),
            dcc.Dropdown(id='select-molecule', options=molecule_options),                       # allow choice among prior names or new name
            html.Div(id='create-molecule'),                                                     # if 'NEW' render text input box here
            html.Br(),
            html.Div(id='select-concentration')                                                 # render appropriate input to select corresponding concentration
        ])

    else:
        return html.Div([

        ])


# @app.callback(Output('dataset-spectrum', 'children'),
#               Input('select-dataset', 'value'))
# def render_dataset_plot(value):
#     if value != None:
#         if value != 'NEW':
#             dataset = dbi.select_dataset(value)
#             return dcc.Graph(
#                 id='spectrum',
#                 figure=px.line(x=dataset.columns.values, y=list(dataset.values[0:3]), labels=['SP_0', 'SP_1', 'SP_2'])
#             )


@app.callback(Output('create-molecule', 'children'),
              Input('select-molecule', 'value'))
def new_molecule_entry(value):
    if value == 'NEW':
        return html.Div(
            dcc.Input(id='new-molecule', type='text', placeholder='')
        )
    else:
        return html.Div([

        ])


@app.callback(Output('select-concentration', 'children'),
              Input('select-molecule', 'value'))                                        # FIXME: when changing selected molecule will not update concentration options
def render_concentration_entry(value):
    if value == 'NEW':
        return html.Div([
            html.Label("Concentration"), 
            dcc.Input(id='new-concentration', type='text', placeholder='')
        ])
    elif value != None:
        print(value)
        concentrations = dataset.loc[value].index.get_level_values('Concentration').unique().to_numpy()
        concentration_options = [{'label': name, 'value': name} for name in concentrations]
        concentration_options.append({'label': 'Create New (enter name below...)', 'value': 'NEW'})
        print(concentration_options)

        return html.Div([
            html.Label("Concentration"), 
            dcc.Dropdown(id='select-concentration', options=concentration_options),
            html.Div(id='create-concentration')
        ])
    else: 
        return html.Div([
            
        ])


@app.callback(Output('create-concentration', 'children'),
              Input('select-concentration', 'value'))
def new_concentration_entry(value):
    if value == 'NEW':
        return html.Div([
            dcc.Input(id='new-concentration', type='text', placeholder='')
        ])
    else:
        return html.Div([

        ])