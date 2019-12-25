# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv("""Path here""")
df2 = pd.read_csv("""Path here""")
df.warehouse_zip = df.warehouse_zip.astype('int').astype('str')
df2.warehouse_zip = df.warehouse_zip.astype('int').astype('str')

opts = [{'label' : i, 'value' : i} for i in df.master_sku.unique()]
warehouse = [{'label' : i, 'value' : i} for i in df.warehouse_zip.unique()]
week = [{'label' : i, 'value' : i} for i in np.array(sorted(df.ds_week.unique()))+1]

trace_1 = go.Bar(x = df[df.master_sku == """Product here"""].ds_week.unique()+1, 
                 y = df[df.master_sku == """Product here"""][df[df.master_sku == """Product here"""].warehouse_zip == """Zipcode here"""].preds,
                 name = 'Prediction')
trace_2 = go.Bar(x = df[df.master_sku == """Product here"""].ds_week.unique()+1, 
                 y = df[df.master_sku == """Product here"""][df[df.master_sku == """Product here"""].warehouse_zip == """Zipcode here"""].actual,
                 name = 'Actual')

layout = go.Layout(title = 'Product Distribution', barmode='group', xaxis_title="Week", yaxis_title="Percentage of Sales")
fig = go.Figure(data = [trace_1, trace_2], layout = layout)
fig.add_trace(go.Scatter(x = df[df.master_sku == """Product here"""].ds_week.unique()+1,
                            y = df[df.master_sku == """Product here"""][df[df.master_sku == """Product here"""].warehouse_zip == """Zipcode here"""].preds,
                            line=dict(dash='dash'),
                            name='Prediction'))
fig.add_trace(go.Scatter(x = df[df.master_sku == """Product here"""].ds_week.unique()+1,
                            y = df[df.master_sku == """Product here"""][df[df.master_sku == """Product here"""].warehouse_zip == """Zipcode here"""].actual,
                            mode='lines',
                            name='Actual'))
#fig.update_layout(barmode='group')

trace_3 = go.Bar(x = df2[df2.master_sku == """Product here"""].ds_week.unique()+1, 
                 y = df2[df2.master_sku == """Product here"""][df2[df2.master_sku == """Product here"""].warehouse_zip == """Zipcode here"""].preds,
                 name = 'Prediction')
trace_4 = go.Bar(x = df2[df2.master_sku == """Product here"""].ds_week.unique()+1, 
                 y = df2[df2.master_sku == """Product here"""][df2[df2.master_sku == """Product here"""].warehouse_zip == """Zipcode here"""].actual,
                 name = 'Actual')

layout2 = go.Layout(title = 'Product', barmode='group', xaxis_title="Week", yaxis_title="Number of Sales")
fig2 = go.Figure(data = [trace_3, trace_4], layout = layout2)

values = df[df.master_sku == """Product here"""][df[df.master_sku == """Product here"""].ds_week == 0].actual
fig3 = go.Figure(data=[go.Pie(labels=df[df.master_sku == """Product here"""][df[df.master_sku == """Product here"""].ds_week == 0].warehouse_zip, values=values)])

fig4 = go.Figure()
color = ['rgba(246, 78, 139, 0.6)', 'rgba(58, 71, 80, 0.6)', 'rgba(120, 60, 200, 0.6)',
         'rgba(58, 71, 80, 0.6)', 'rgba(23, 173, 50, 0.6)', 'rgba(185, 71, 12, 0.6)',
         'rgba(71, 60, 29, 0.6)', 'rgba(15, 171, 95, 0.6)', 'rgba(98, 17, 180, 0.6)']
for i in range(len(df2[df2.master_sku == """Product here"""].warehouse_zip.unique())):
    fig4.add_trace(go.Bar(
        y=df2[df2.master_sku == """Product here"""][df2[df2.master_sku == """Product here"""].warehouse_zip == df2[df2.master_sku == """Product here"""].warehouse_zip.unique()[i]].actual,
        x=df2[df2.master_sku == """Product here"""][df2[df2.master_sku == """Product here"""].warehouse_zip == df2[df2.master_sku == """Product here"""].warehouse_zip.unique()[i]].ds_week.unique()+1,
        name=df2[df2.master_sku == """Product here"""].warehouse_zip.unique()[i],
        marker=dict(
            color=color[i]
            )
        ))

fig4.update_layout(barmode='stack')





app.layout = html.Div([
    # adding a plot
    dcc.Graph(id = 'plot', figure = fig),
    # dropdown
    html.P([
        html.Label("Choose a Product"),
        dcc.Dropdown(id = 'opt', options = opts,
                     value = opts[0]['value']),

        html.Label("Choose a warehouse"),
        dcc.Dropdown(id = 'opt2', options = warehouse,
                     value = warehouse[0]['value'])
        ], style = {'width': '400px',
                    'fontSize' : '20px',
                    'padding-left' : '100px',
                    'display': 'inline-block'}),
    # adding a plot
    dcc.Graph(id = 'plot2', figure = fig2),
    # dropdown
    html.P([
        html.Label("Choose a Product"),
        dcc.Dropdown(id = 'opt3', options = opts,
                     value = opts[0]['value']),

        html.Label("Choose a warehouse"),
        dcc.Dropdown(id = 'opt4', options = warehouse,
                     value = warehouse[0]['value'])
        ], style = {'width': '400px',
                    'fontSize' : '20px',
                    'padding-left' : '100px',
                    'display': 'inline-block'}),

    # adding a plot
    dcc.Graph(id = 'plot3', figure = fig3),
    # dropdown
    html.P([
        html.Label("Choose a Product"),
        dcc.Dropdown(id = 'opt5', options = opts,
                     value = opts[0]['value']),

        html.Label("Choose a week"),
        dcc.Dropdown(id = 'opt6', options = week,
                     value = week[0]['value'])
        ], style = {'width': '400px',
                    'fontSize' : '20px',
                    'padding-left' : '100px',
                    'display': 'inline-block'}),

    # adding a plot
    dcc.Graph(id = 'plot4', figure = fig4),
    # dropdown
    html.P([
        html.Label("Choose a feature"),
        dcc.Dropdown(id = 'opt7', options = opts,
                     value = opts[0]['value'])
        ], style = {'width': '400px',
                    'fontSize' : '20px',
                    'padding-left' : '100px',
                    'display': 'inline-block'})

    ])
        
@app.callback(Output('plot', 'figure'),
             [Input('opt', 'value'),
              Input('opt2', 'value')])

def update_figure(X, Y):
    data = df[df.master_sku == X][df[df.master_sku == X].warehouse_zip == Y]
    trace_1 = go.Bar(x = df[df.master_sku == X].ds_week.unique()+1, y = data.preds,
                     name = 'Prediction')
    trace_2 = go.Bar(x = df[df.master_sku == X].ds_week.unique()+1, y = data.actual,
                     name = 'Actual')
    fig = go.Figure(data = [trace_1, trace_2], layout = layout)
    fig.add_trace(go.Scatter(x = df[df.master_sku == X].ds_week.unique()+1,
                                y = df[df.master_sku == X][df[df.master_sku == X].warehouse_zip == Y].preds,
                                line=dict(dash='dash'),
                                name='Prediction'))
    fig.add_trace(go.Scatter(x = df[df.master_sku == X].ds_week.unique()+1,
                                y = df[df.master_sku == X][df[df.master_sku == X].warehouse_zip == Y].actual,
                                mode='lines',
                                name='Actual'))
    #fig.update_layout(barmode='group')
    return fig

@app.callback(Output('plot2', 'figure'),
             [Input('opt3', 'value'),
              Input('opt4', 'value')])

def update_figure2(X, Y):
    data = df2[df2.master_sku == X][df2[df2.master_sku == X].warehouse_zip == Y]
    trace_3 = go.Bar(x = df2[df2.master_sku == X].ds_week.unique()+1, y = data.preds,
                     name = 'Prediction')
    trace_4 = go.Bar(x = df2[df2.master_sku == X].ds_week.unique()+1, y = data.actual,
                     name = 'Actual')
    fig = go.Figure(data = [trace_3, trace_4], layout = layout2)
    #fig.update_layout(barmode='group')
    return fig

@app.callback(Output('plot3', 'figure'),
             [Input('opt5', 'value'),
              Input('opt6', 'value')])

def update_figure3(X, Y):
    data = df[df.master_sku == X][df[df.master_sku == X].ds_week == Y-1]
    values = data.actual
    fig = go.Figure(data=[go.Pie(labels=data.warehouse_zip, values=values)])

    return fig

@app.callback(Output('plot4', 'figure'),
             [Input('opt7', 'value')])

def update_figure4(X):
    fig4 = go.Figure()
    color = ['rgba(246, 78, 139, 0.6)', 'rgba(58, 71, 80, 0.6)', 'rgba(120, 60, 200, 0.6)',
             'rgba(58, 71, 80, 0.6)', 'rgba(23, 173, 50, 0.6)', 'rgba(185, 71, 12, 0.6)',
             'rgba(71, 60, 29, 0.6)', 'rgba(15, 171, 95, 0.6)', 'rgba(98, 17, 180, 0.6)']
    for i in range(len(df2[df2.master_sku == X].warehouse_zip.unique())):
        fig4.add_trace(go.Bar(
            y=df2[df2.master_sku == X][df2[df2.master_sku == X].warehouse_zip == df2[df2.master_sku == X].warehouse_zip.unique()[i]].actual,
            x=df2[df2.master_sku == X][df2[df2.master_sku == X].warehouse_zip == df2[df2.master_sku == X].warehouse_zip.unique()[i]].ds_week.unique()+1,
            name=df2[df2.master_sku == X].warehouse_zip.unique()[i],
            marker=dict(
                color=color[i]
                )
            )) 
    fig4.update_layout(barmode='stack')

    return fig4

if __name__ == '__main__':
    app.run_server(debug=True)
