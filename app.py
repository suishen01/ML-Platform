import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv(
    'test_data_for_app.csv')

app.layout = html.Div([
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': '12994', 'value': '12994'},
            {'label': '19049', 'value': '19049'}
        ],
        value='12994'
    ),
    html.Div(id='output'),

])

@app.callback(Output('output', 'children'), [Input('my-dropdown', 'value')])
def display_graphs(selected_values):
    graphs = []
    for values in selected_values:
        graphs.append(html.Div(dcc.Graph(
            id='ret-vs-date',
            figure={
                'data': [
                    go.Scatter(
                        x=df['date'],
                        y=df[df['gvkey'].astype('str') == selected_values]['ret'],
                        text=df[df['gvkey'].astype('str') == selected_values]['gvkey'],
                        mode='lines+markers',
                        opacity=0.7,
                        marker={
                            'size': 1,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=selected_values
                    ),
                    go.Scatter(
                        x=df['date'],
                        y=df[df['gvkey'].astype('str') == selected_values]['predicted_ret'],
                        text=df[df['gvkey'].astype('str') == selected_values]['gvkey'],
                        mode='lines+markers',
                        opacity=0.7,
                        marker={
                            'size': 1,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=selected_values + '(predicted)'
                    )
                ],
                'layout': go.Layout(
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Ret'},
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                    legend={'x': 0, 'y': 1},
                    hovermode='closest'
                )
            }
        )))
    return graphs

if __name__ == '__main__':
    app.run_server(debug=True)
