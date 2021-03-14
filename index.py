import dash_html_components as html
import dash_core_components as dcc

from dash.dependencies import Input, Output
from app import app
from app import server

import database_interface as dbi

### NOTE:TEMPORARY, have external function handle
import pandas as pd

app.layout = html.Div([
        dcc.Tabs(
            id='tab-index', 
            value='tab-1', 
            children=[
                dcc.Tab(label='Import New Data', value='tab-1'),
                dcc.Tab(label='Train Model', value='tab-2'),
                dcc.Tab(label='Test Model', value='tab-3'),
                ]
        ),
        html.Div(id='tab-content')
    ], style={"background": "#ffffff"}
)

@app.callback(Output('tab-content', 'children'),
              Input('tab-index', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        dataset_options = [{'label': name, 'value': name} for name in dbi.list_datasets()]
        dataset_options.append({'label': 'Create New (enter name below...)', 'value': 'NEW'})

        return html.Div([

            html.Label("Dataset"),
            dcc.Dropdown(
                id='dataset-selection',
                options=dataset_options,
                placeholder='Select a Dataset...'
            ),

            html.Div(id='dataset-entry'),

            dcc.Graph(
                id='graph-1',
                figure={
                    'data': [
                        {'x': [1, 2, 3, 4, 5, 6, 7], 'y': [10, 20, 30, 40, 50, 60, 70], 'type': 'line', 'name': 'value1'},
                        {'x': [1, 2, 3, 4, 5, 6, 7], 'y': [12, 22, 36, 44, 49, 58, 73], 'type': 'line', 'name': 'value2'}
                    ],
                    'layout': {'title': 'Simple Line Graph',}
                }
            ),
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab content 2')
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Tab content 3')
        ])


@app.callback(Output('dataset-entry', 'children'),
              Input('dataset-selection', 'value'))
def render_dataset_entry(value):
    if value == 'NEW':
        return html.Div([
            dcc.Input(id='dataset-name', type='text', placeholder=''),
            html.Br(),
            html.Label("Molecule"),
            dcc.Input(id='molecule', type='text', placeholder=''),
            html.Br(),
            html.Label("Concentration"),
            dcc.Input(id='concentration', type='text', placeholder='')
        ])
        
    elif value != None:
        dataset = dbi.select_dataset(value)

        molecules = dataset.index.get_level_values('Molecule').unique().to_numpy()
        molecule_options = [{'label': name, 'value': name} for name in molecules]
        molecule_options.append({'label': 'Create New (enter name below...)', 'value': 'NEW'})

        concentrations = dataset.index.get_level_values('Concentration').unique().to_numpy()
        concentration_options = [{'label': name, 'value': name} for name in concentrations]
        concentration_options.append({'label': 'Create New (enter name below...)', 'value': 'NEW'})

        return html.Div([
            html.Label("Molecule"),
            dcc.Dropdown(id='prev-molecule', options=molecule_options),
            html.Div(id='new-molecule'),
            html.Br(),
            html.Label("Concentration"), 
            dcc.Dropdown(id='prev-concentration', options=concentration_options),
            html.Div(id='new-concentration')
        ])

    else:
        return html.Div([

        ])


@app.callback(Output('new-molecule', 'children'),
              Input('prev-molecule', 'value'))
def new_molecule_option(value):
    if value == 'NEW':
        html_element = html.Div(
            dcc.Input(id='new_molecule', type='text', placeholder='')
        )
    else:
        html_element = html.Div()
    return html_element


@app.callback(Output('new-concentration', 'children'),
              Input('prev-concentration', 'value'))
def new_concentration_option(value):
    if value == 'NEW':
        html_element = html.Div(
            dcc.Input(id='new_concentration', type='text', placeholder='')
        )
    else:
        html_element = html.Div()
    return html_element

if __name__ == '__main__':
    app.run_server(debug=True)