import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_html_components as html
from dash.exceptions import PreventUpdate
from operator import add

import dash
import pandas as pd
import pathlib
import numpy as np
import pandas as pds
import plotly.graph_objects as go
import plotly.express as px

from app import app
from dependencies import CONTENT_STYLE

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
offset = [
             0] * 3  # set to zero in order to have a starting value # TODO: this must be hard coded and saved somewhere in the data

# ==============================    Settings   ==============================
width_subplot = 300
height_subplot = 250
marker_colors = ['#8F5C32', '#D0D4A3', '#888F24']  # colors derived from 'main color'=#2B3C8F  (blueish)
line_colors = ['#2B3C8F', '#4A4942']
fig_all = go.Figure()
# ==============================    Data   ==============================
# rotation = rotation_all['left']

dfv = pd.read_csv(DATA_PATH.joinpath("vgsales.csv"))  # GregorySmith Kaggle
sales_list = ["North American Sales", "EU Sales", "Japan Sales", "Other Sales", "World Sales"]


# ------------------------- Plotting part  ------------------------- #
def plot_intensities(intensity, valleys, title_id, fft_data='', peaks='', width=width_subplot, height=height_subplot):
    """ visualises data intensity at all 360 degrees around the marker"""

    # ====================    get data into a frame to facilitate plotting   ====================
    angles = intensity[:, 1]
    if title_id == 'intensity-profile-marker':
        data2plot = pds.DataFrame([intensity[:, 0], fft_data], index=["intensities", "fft"],
                                  columns=np.rad2deg(angles)).T
    else:
        data2plot = pds.DataFrame([intensity[:, 0]], index=["intensities"], columns=np.rad2deg(angles)).T

    data2plot['id'] = data2plot.index
    data2plot = pds.melt(data2plot, id_vars='id', value_vars=data2plot.columns[:-1])

    # ================    line plot of intensities (and estimated FFT in the case of 'marker')    ================
    if title_id == 'intensity-profile-marker':
        fig = px.line(data2plot, x='id', y='value', width=width, height=height, color='variable',
                      color_discrete_map={
                          "intensities": line_colors[0],
                          "fft": line_colors[1]})
    else:
        fig = px.line(data2plot, x='id', y='value', width=width, height=height, color='variable',
                      color_discrete_map={"intensities": line_colors[0]})

    fig.add_hline(y=data2plot.min()['value'] * 1.05, line_width=1.2)  # somehow bulky hack for x/y-axis
    fig.add_vline(x=0, line_width=1.2)
    fig.update_traces(line=dict(width=1.0), showlegend=False)

    fig.update_yaxes(showticklabels=False, showgrid=False, title_text="", zerolinewidth=1.5, zeroline=True)
    fig.update_xaxes(showticklabels=True, showgrid=True, tickvals=np.arange(0, 361, step=60), gridwidth=.25,
                     gridcolor='lightgray', title_text="", ticks='inside')
    fig.add_trace(go.Scatter(x=[np.rad2deg(angles)], y=[fft_data], name='fft', mode='lines', showlegend=False))
    fig.update_xaxes()

    # ====================    display all valleys (and peaks for marker)   ====================
    if title_id == 'intensity-profile-marker':
        for peak, v in zip(peaks, valleys):
            fig.add_trace(go.Scatter(x=[np.rad2deg(angles[int(peak)])], y=[intensity[int(peak), 0]],
                                     mode='markers', showlegend=False, marker=dict(size=8, color=marker_colors[0],
                                                                                   symbol='cross')))
            fig.add_trace(go.Scatter(x=[np.rad2deg(angles[int(v)])], y=[intensity[int(v), 0]],
                                     mode='markers', showlegend=False, marker=dict(size=8, color=marker_colors[1],
                                                                                   symbol='cross')))
    else:
        for v in valleys:
            fig.add_trace(go.Scatter(x=[np.rad2deg(angles[int(v)])], y=[intensity[int(v), 0]],
                                     mode='markers', showlegend=False, marker=dict(size=8, color=marker_colors[0],
                                                                                   symbol='cross')))

    fig.update_layout(title_text="Intensity profile", title_x=.5, title_y=.95,
                      margin=dict(l=20, r=20, t=45, b=10), paper_bgcolor='white', plot_bgcolor='white',
                      width=width, height=height)

    return fig


def plotCTintensities(intensity_matrix, fitvolume, data_surface):
    """function plotting perpendicular images of the intensities obtained from trajectory coordinates; separate
    function needed here as changes occur in trejectories/markers"""

    for idx, (direction, intensity) in enumerate(intensity_matrix.items()):
        matrix2plot = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))  # Prepare data!
        X, Y, Z = fitvolume[direction][0, :, :], fitvolume[direction][1, :, :], fitvolume[direction][2, :, :]
        slice_temp = get_the_slice(X, Y, Z, matrix2plot)
        data_surface.append(slice_temp)

    return data_surface


def plot_coordinates(coordinates):
    """plots coordinates of markers to the surface plots generated so far and returns graph object (go)"""

    plot_content = []
    xs, ys, zs = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    for x, y, z in zip(xs,ys,zs):
        trace = go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode="markers",
            showlegend=False,
            name="",
            marker=dict(
                symbol="circle-open",
                size=15,
                color='#000000'))
        plot_content.append(trace)

    return plot_content


def get_the_slice(x, y, z, surfacecolor, colorscale='Greys', showscale=False):
    """ returns a 'slice' of 3D image as a graph object (go)"""
    return go.Surface(x=x,  # https://plot.ly/python/reference/#surface
                      y=y,
                      z=z,
                      surfacecolor=surfacecolor,
                      colorscale=colorscale,
                      showscale=showscale,
                      name="")


# -------------------------          Data visualisation   ------------------------- #
layout_container = html.Div(id='visualisation-container', children=[
    html.H1('Please select a subject to display', id='headline-tab-visualisation',
            style={"margin-upper": "15px", "textAlign": "center"}),
    # First row with 3D view in anterior-posterior direction
    dbc.Row(
        [
            dcc.Loading(
                id="loading-marker-axial",
                type="default",
                children=dcc.Graph(id='axialslice-first', figure={})),
        ]),

    # Second row with 3D view in lateral-medial direction
    # html.Div(offset, id='intermediate-value', style={'display': 'none'}),  # to store offset as this may be modified
    # html.Div(offset, id='loaded_data', style={'display': 'none'})  # to store offset as this may be modified
])


#  =================== Callback returning headline for tab ===================
@app.callback(dash.dependencies.Output('headline-tab-visualisation', 'children'),
              dash.dependencies.Input('hemisphere-headline', 'children'))
def headline_visualisation(value):
    """changes headline according to pressed button on sidebar"""
    if value is None:
        term = 'Please select Subject'
    else:
        term = 'Visualisation of {}'.format(value)

    return term


#  =================== Callback returning slices at first angulation ===================
@app.callback(dash.dependencies.Output('axialslice-first', 'figure'), # TODO: id-naming is misleading!
              dash.dependencies.Input('results-hemisphere', 'data'))
def axialslices_plotting(value):
    """creates graphical objects for similarities at level1/2 for plotting purposes"""
    layout_plot = go.Layout(width=600, height=500, autosize=True,
                            margin=go.layout.Margin(l=5, r=5, b=5, t=5, pad=0),
                            scene=dict(
                                xaxis_visible=False,
                                yaxis_visible=False,
                                zaxis_visible=False,
                                camera=dict(
                                    center=dict(x=0, y=0, z=0),
                                    eye=dict(x=0, y=-1.2, z=0),
                                    up=dict(x=0, y=0, z=1)
                                )
                            )
                            )
    if value is None:
        go_intensities = [dict() for _ in range(1)]
        print('none selected')
    else:
        print('something selected')
        data_surface = plot_coordinates(value['coordinates'])
        go_intensities = plotCTintensities(value['intensities'],
                                           value['fitvolumes'],
                                           data_surface)

    # TODO: add the second visualisation here and modify the if value ... part

    fig = {'data': go_intensities, 'layout': layout_plot}

    return fig
