import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
import pandas as pd
import base64
import io

from app import app
from app import server
import database_interface as dbi
import data_analysis as da

def render_tab():
    dataset_options = [{'label': name, 'value': name} for name in dbi.list_datasets()]
    dataset_options.append({'label': 'Create New (enter name below...)', 'value': 'NEW'})

    return html.Div([
        dcc.Location(id='url', refresh=True),           # FIXME: might be improper implementation of page refresh?

        dbc.Row([
            dbc.Col([
                html.H3("New Data Import"),
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

                html.Div([
                    dcc.Dropdown(
                        id='select-molecule',
                        placeholder='Select a Molecule...'
                    )
                ],
                    id='select-molecule-container',
                    style={'display': 'none'}
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

                html.Div([
                    dcc.Dropdown(
                        id='select-concentration',
                        placeholder='Select a Concentration...'
                )
                ],
                    id='select-concentration-container',
                    style={'display': 'none'}
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
                    multiple=False
                ),

                html.Div(id='import-data', style={'display':'none'}),       # hidden div for imported file converted to df

                dcc.Checklist(
                    id='remove-unselected',
                    options=[
                        {'label': 'Remove Unselected Spectra', 'value': 'remove'}
                    ],
                    value=[]
                ),

                html.Button(
                    'Import',
                    id='submit-import',
                    n_clicks=0
                ),

                html.H5(id='intermediate-value')    # status of import success
            ],
                style={
                    'background-color': 'lightgrey',
                    'padding': '25px'
                },
                width=3
            ),

            dbc.Col([
                html.Div(
                    [
                        html.H3('Import Spectra Visualizer/Editor'),
                        dcc.Graph(
                            id='import-spectra'
                        )
                    ],
                    style={
                        'background-color': 'LightSalmon',
                        'padding': '25px',
                    },
                    id='import-spectra-container'
                ),

                html.Div(id='unselected-data', style={'display':'none'})   # hidden div for tracking spectra to remove
            ])
        ])
    ])

@app.callback(Output('new-dataset-container', 'style'),
              Input('select-dataset', 'value'))
def update_dataset_input(value):
    enabled_style = {'display': 'block'}
    disabled_style = {'display': 'none'}

    # check if defining new dataset, if so show textbox for entry
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

    # check if defining new dataset
    if dataset_value == 'NEW':
        return options, disabled_style, enabled_style
    # otherwise check if dataset choice is selected
    elif dataset_value != None:
        dataset = dbi.select_dataset(dataset_value)

        molecules = dataset.index.get_level_values('Molecule').unique().to_numpy()              # list unique molecule labels within dataset
        options = [{'label': name, 'value': name} for name in molecules]
        options.append({'label': 'Create New (enter name below...)', 'value': 'NEW'})           # allow choice to assign new molecule label

        # if defining new molecule, serve textbox for entry 
        if molecule_value == 'NEW':
            return options, enabled_style, enabled_style
        # if other molecule label selected, do not show textbox
        elif molecule_value != None:
            return options, enabled_style, disabled_style
        # if no selection for molecule, do now show textbox yet
        else:
            return options, enabled_style, disabled_style

    # if not, wait to display proper molecule and concentration options
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

    # check if defining new dataset
    if dataset_value == 'NEW':
        return options, disabled_style, enabled_style

    # otherwise check if selections for dataset and molecule labels 
    elif dataset_value != None and molecule_value != None:
        # if new molecule, serve new concentration dialog
        if molecule_value == 'NEW':
            return options, disabled_style, enabled_style
        # otherwise, serve dialog containing dropdown list of existing concentrations
        else:
            dataset = dbi.select_dataset(dataset_value)

            molecules = dataset.loc[molecule_value].index.get_level_values('Concentration').unique().to_numpy()              # list unique molecule labels within dataset
            options = [{'label': name, 'value': name} for name in molecules]
            options.append({'label': 'Create New (enter name below...)', 'value': 'NEW'})  # allow choice to assign new molecule label

            # if user selects to define new concentration, serve textbox
            if concentration_value == 'NEW':
                return options, enabled_style, enabled_style

            # if selection is some other value, don't serve textbox
            elif concentration_value != None:
                return options, enabled_style, disabled_style

            # otherwise, no selection and hide textbox for now
            else:
                return options, enabled_style, disabled_style

    # otherwise no molecule selected, wait to display proper concentration dialog
    else:
        return options, disabled_style, disabled_style


def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            # assume that the user uploaded comma seperated TXT file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df

@app.callback(Output('import-data', 'children'),
              Output('unselected-data', 'children'),
              Input('upload-data', 'contents'),
              Input('upload-data', 'filename'),
              Input('import-spectra', 'restyleData'),
              State('import-data', 'children'),
              State('unselected-data', 'children'))
def update_import_data(contents, filename, style_data, import_data, unselected_data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    # check if new file uploaded, if so store in hidden div
    if 'upload-data' in changed_id:
        contents = contents
        filename = filename
        df = parse_data(contents, filename)
        unselected_data = []
        return df.to_json(), unselected_data
    
    # otherwise check if changes to selected traces, update set of unselected traces
    elif 'import-spectra' in changed_id:
        event_states = style_data[0]['visible']
        event_idxs = style_data[1]
        for idx,state in zip(event_idxs, event_states):
            if state != True and idx not in unselected_data:
                unselected_data.append(idx)
            elif idx in unselected_data:
                while idx in unselected_data: unselected_data.remove(idx)
        return dash.no_update, unselected_data
    else:
        return dash.no_update, dash.no_update
    

@app.callback(Output('import-spectra', 'figure'),
              Input('import-data', 'children'))
def update_graph(import_data):
    if import_data:
        df = pd.read_json(import_data)
        x_axis = df['Raman Shift']
        df.drop('Raman Shift',axis=1, inplace=True)

        fig = go.Figure()
        fig.update_xaxes(title_text='Raman Shift cm^-1')
        fig.update_yaxes(title_text='Relative Intensity')

        for signal in df.columns:
            y_axis = df[signal]
            fig.add_trace(go.Scatter(x=x_axis, y=y_axis, name=signal))
        
        return fig

@app.callback(Output('intermediate-value', 'children'),
              Output('url', 'href'),
              Input('submit-import', 'n_clicks'),
              State('select-dataset', 'value'),
              State('new-dataset', 'value'),
              State('select-molecule', 'value'),
              State('new-molecule', 'value'),
              State('select-concentration', 'value'),
              State('new-concentration', 'value'),
              State('import-data', 'children'),
              State('unselected-data', 'children'),
              State('remove-unselected', 'value'))
def import_data(n_clicks, select_dataset, new_dataset, select_molecule, new_molecule, select_concentration, new_concentration, import_data, unselected_data, remove_unselected):

    df = pd.DataFrame()
    dataset_label = None
    molecule_label = None
    concentration_label = None

    # check all conditions for ability to upload new data
    if select_dataset == 'NEW':
        dataset_label = new_dataset
        molecule_label = new_molecule
        concentration_label = new_concentration
    elif select_molecule == 'NEW':
        dataset_label = select_dataset
        molecule_label = new_molecule
        concentration_label = new_concentration
    elif select_concentration == 'NEW':
        dataset_label = select_dataset
        molecule_label = select_molecule
        concentration_label = new_concentration
    else:
        dataset_label = select_dataset
        molecule_label = select_molecule
        concentration_label = select_concentration

    dataset = dbi.select_dataset(dataset_label)         # should get either exisiting or empty dataframe

    # get df from selected file to import
    if import_data:
        df = pd.read_json(import_data)
        if remove_unselected:
            unselected_data = [val+1 for val in unselected_data]
            df = df.drop(df.columns[unselected_data], axis=1)

    # ensure all necessary labels/files are defined for import
    if dataset_label!=None and molecule_label!=None and concentration_label!=None and not df.empty:
        dataset = da.append_to_dataset(dataset, df, molecule_label, concentration_label)
        dbi.store_dataset(dataset, dataset_label)
        return 'Successful', '/'
    else:
        return 'No Import Yet', dash.no_update