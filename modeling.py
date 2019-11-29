import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import pickle
import numpy as np
from sklearn.preprocessing import FunctionTransformer


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
supermarket = pd.read_csv('supermarket_sales - Sheet1.csv')
lm = pickle.load(open('lm.sav','rb'))
log_transformer = FunctionTransformer(np.log,np.exp)

app.layout = html.Div(children = [
            dcc.Tabs(value='tabs', id='tabs-1',children=[
                    dcc.Tab(label='Home',children = [
                        html.Div(children = [
                            html.Div(children = [
                                    html.Center(html.H1('SUPERMARKET DATA')),
                                    html.Center(html.H2('by: Ina', style = {'font-weight' : 'bold', 'font-style' : 'italic'})),
                                    html.Div(html.Img(src = '/assets/images.jpg'),
                                            style ={'text-align' : 'center', 'margin-top' : '80px',
                                                    'margin-bottom' : '80px'}),

                                    html.Div(id = 'tabel',
                                        children = [dash_table.DataTable(id='data_table',
                                        columns=[{"name": i, 'id' : i} for i in supermarket.columns],
                                        data = supermarket.to_dict('records'),
                                        sort_action='native',
                                        filter_action='native',
                                        page_action = 'native',
                                        page_current = 0,
                                        page_size = 10,
                                        style_table = {'overflowX': 'scroll'})]
                                    )
                                ], className='card-body'),
                            ], className='card'),
                    ],className='col-2'),

                    dcc.Tab(
                        label = 'Forecast',
                        children=[
                        html.Div(children=[
                            html.Div(children=[
                                html.P('Unit price'),
                                html.Div(children=[
                                    dcc.Input(
                                        id='unit-price',
                                        type='number',
                                        min=1,max=1000,
                                        step=1
                                        
                                )])
            
                        ],className='col-3'),

                            html.Div(children=[
                                html.P('Rating'),
                                html.Div(children = [
                                    dcc.Input(
                                        id='rating',
                                        type='number',
                                        min=1,max=10,
                                        step=1
                
                                    )
                                ])
                            ],className='col-3')
                        ],className='row'),

                        html.Div(children=[
                            html.Div(children=[
                                html.P('Quantity'),
                                html.Div(children = [
                                    dcc.Input(
                                        id='quantity',
                                        type='number',
                                        min=1,max=10,
                                        step=1
                                    )
                                ])
                            ],className='col-3'),

                            html.Div(children=[
                                html.P('City'),
                                html.Div(children = [
                                    dcc.Dropdown(
                                        id='city',
                                        options = [{'label': i, 'value':i} for i in supermarket['City'].unique()],
                                    )
                                ])
                            ],className='col-3'),
                        ],className='row'),

                        html.Div(html.Button('Search', id = 'prediksi'),className='col-3'),
                        html.Div(id='hasil')
                    ],className='col-2'),
                   
                    ]),
                    
], 
style = {
    'maxWidth' : '1200px',
    'margin' : '0 auto',
    })   



@app.callback(
    Output(component_id='hasil',component_property='children'),
    [Input(component_id='prediksi',component_property='n_clicks')],
    [State(component_id='unit-price',component_property='value'),
    State(component_id='rating',component_property='value'),
    State(component_id='quantity',component_property='value'),
    State(component_id='city',component_property='value')
    ]
)

def hasil (n_clicks,unitprice,rating,quantity,city):
    if city == 'Naypyitaw':
        city_naypyitaw = 1
        city_yangon = 0
    elif city == 'Yangon' :
        city_naypyitaw = 0
        city_yangon = 1
    else :
        city_naypyitaw = 0
        city_yangon = 0

    if n_clicks == None:
        return 'Prediksi Hasil'

    else:
        unitprice = log_transformer.transform(np.array(unitprice).reshape(1,-1))
        prediksi = lm.predict(np.array([unitprice,rating,quantity,city_naypyitaw,city_yangon]).reshape(1,-1))
        prediksi = log_transformer.inverse_transform(np.array(prediksi).reshape(1,-1))[0][0]

        return str(prediksi)

                    

if __name__ == '__main__':
    app.run_server(debug=True)
