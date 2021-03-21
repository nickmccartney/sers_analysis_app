import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px

from dash.dependencies import Input, Output
from app import app
from app import server

import database_interface as dbi
import tab_1

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
        return tab_1.render_tab()

    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab content 2')
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Tab content 3')
        ])




# must have submit button callback 
## consider input from new-concentration, select-molecule, new-molecule, select-concentration, new-concentration id's


if __name__ == '__main__':
    app.run_server(debug=True)