import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import pathlib
import os
import numpy as np
from app import app
from datasets.leadRotation import PrepareData
import plotly.graph_objects as go
import plotly.express as px

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

# Get data for the contents of the GUI
print("Extracting rotation ...", flush=True)
inputfolder = '/media/storage/cDBS/data/NIFTI/' # TODO: this must be included somehow somewhere
subject = 'subj3'
p = PrepareData()
angles, slices, intensities_marker_angle, intensities_level_angle, sum_intensity, roll, dir_valleys, markerfft, valleys, peaks = \
    p.getData(subj=subject, inputfolder=inputfolder)
print("Done", end ='')


dfv = pd.read_csv(DATA_PATH.joinpath("vgsales.csv"))  # GregorySmith Kaggle
sales_list = ["North American Sales", "EU Sales", "Japan Sales", "Other Sales",	"World Sales"]

def prepare_imshow(data, title, center='', color_scale='blues', width=300, height=300, clims=(-1024, 4096)):
    data_interp = np.interp(data, (data.min(), data.max()), clims)
    id = '{}-rotation'.format(title).lower()
    fig = px.imshow(data_interp, color_continuous_scale=color_scale, width=width, height=height)
    if center:
       fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], mode='markers', marker=dict(size=10, color='green'),
                                marker_symbol='circle-open'))
    fig.update_layout(coloraxis_showscale=False, title_text=title, title_x=.7, title_y=.92,
                      margin=dict(l=80, r=20, t=15, b=10))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return dcc.Graph(id=id, figure=fig)

def plot_intensities(intensities, markerfft, angles, peaks, valleys, id, width=300, height=300):

    fig = px.line(x=np.rad2deg(angles), y=intensities, width=width, height=height, template='none')
    fig.add_trace(go.Scatter(x=[np.rad2deg(angles)], y=[markerfft], name='fft', mode='lines', showlegend=False))
    for p,v in zip(peaks, valleys):
        fig.add_trace(go.Scatter(x=[np.rad2deg(angles[int(p)])], y=[intensities[int(p)]],
                                 mode='markers', showlegend=False, marker=dict(size=10, color='green')))
        fig.add_trace(go.Scatter(x=[np.rad2deg(angles[int(v)])], y=[intensities[int(v)]],
                                 mode='markers', showlegend=False, marker=dict(size=10, color='red')))

    fig.update_layout(title_text="Intensity profile", title_x=.7, title_y=.92,
                      margin=dict(l=80, r=20, t=15, b=10), paper_bgcolor='white', plot_bgcolor='white',
                      width=width, height=height)
    fig.update_yaxes(showticklabels=False, showgrid=False, title_text="", showline=True,
                   linewidth=1.5, zeroline=False)
    fig.update_xaxes(showticklabels=True, showgrid=False, tickvals=list(range(0,400,100)),
                     title_text="", showline=True, ticks='outside', linewidth=1.5, zeroline=False)

    return dcc.Graph(id=id, figure=fig)

layout_container = html.Div(id='rotation-container', children=[
    html.H1("Rotation: Left hemisphere", style={"margin-upper": "15px", "textAlign": "center"}),
    dbc.Row([
        dbc.Col(prepare_imshow(slices['marker'], title='Marker', center=tuple(int(round(s/2)) for s in slices['marker'].shape)), width=4),
        dbc.Col(plot_intensities(intensities_marker_angle, markerfft, angles, peaks, valleys, 'intensity-profile-marker'), width=4)]),
    dbc.Row([
        dbc.Col(prepare_imshow(slices['level1'], title='Level1', center=tuple(int(round(s/2)) for s in slices['marker'].shape))),
        dbc.Col()]),
    dbc.Row([
        dbc.Col(prepare_imshow(slices['level2'], title='Level2', center=tuple(int(round(s/2)) for s in slices['marker'].shape))),
        dbc.Col()])
])

#@app.callback(
#    Output(component_id='marker_rotation', component_property='figure'),
#    [Input(component_id='genre-dropdown', component_property='value'),
#     Input(component_id='sales-dropdown', component_property='value')]
#)
#def display_marker_rotation(genre_chosen, sales_chosen):
#    layout_plot = go.Layout(width=600, height=500, yaxis=go.layout.YAxis(visible=False, range=[0, 120]),
#                        margin=go.layout.Margin(l=5, r=5, b=5, t=5, pad=0))
#    fig = {'data': px.imshow(slices['marker']), 'layout': layout_plot}
#
#    return fig
