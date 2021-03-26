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

from app import app #, rotation_all
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

# ==============================    Data   ==============================
# rotation = rotation_all['left']

dfv = pd.read_csv(DATA_PATH.joinpath("vgsales.csv"))  # GregorySmith Kaggle
sales_list = ["North American Sales", "EU Sales", "Japan Sales", "Other Sales", "World Sales"]


# ------------------------- Plotting part  ------------------------- #
def prepare_imshow(data, valleys, xy_box, point_of_interest, title='Marker', redirect=True,
                   width=width_subplot, height=height_subplot, color_scale='ice', clims=(-50, 100), radius=20):
    """ this part plots the intensities of the marker and the different levels at which rotation is estimated;
    coors are derived from the blue-tone #2B3C8F"""

    if redirect:
        data = np.fliplr(data)
        valleys = np.multiply(valleys - 360, -1)

    # ====================    display intensities at axial slices and show annotations   ====================
    fig_left = px.imshow(data.T, x=xy_box[0, :], y=xy_box[1, :], color_continuous_scale=color_scale,
                         width=width, height=height, range_color=clims, aspect='equal', origin='lower')
    directions = dict({'P': [point_of_interest[0], np.percentile(xy_box[1, :], 8)],
                       'A': [point_of_interest[0], np.percentile(xy_box[1, :], 92)],
                       'R': [np.percentile(xy_box[0, :], 8), point_of_interest[1]],
                       'L': [np.percentile(xy_box[0, :], 92), point_of_interest[1]]})

    for ann, value in directions.items():
        fig_left.add_annotation(dict(x=value[0], y=value[1],
                                     showarrow=False, text=ann, xanchor='center', font=dict(size=20, color='white')))

    # ====================    add coordinates at center   ====================
    fig_left.add_trace(go.Scatter(x=[point_of_interest[0]], y=[point_of_interest[1]], mode='markers',
                                  marker_symbol='circle-open', marker=dict(size=12, color=marker_colors[0])))

    # ====================    add marker/lines for estimated angles   ====================
    factor = 1.2 if title == 'Marker' else 1.0
    vector = vector_intensity(point_of_interest, pixdim=[.7] * 3, radius=16)  # creates vector wrt center

    fig_left.add_shape(type='circle', xref="x", yref="y",
                       x0=point_of_interest[0] - radius / 2, y0=point_of_interest[1] - radius / 2,
                       x1=point_of_interest[0] + radius / 2, y1=point_of_interest[1] + radius / 2,
                       line=dict(
                           color=marker_colors[0],
                           width=.75,
                           dash='dot'))
    for val in valleys:
        fig_left.add_shape(type='line',
                           x0=point_of_interest[0],  # detected marker/center of level1/2
                           y0=point_of_interest[1],
                           x1=point_of_interest[0] + factor * (vector[int(val)][0] - point_of_interest[0]),
                           y1=point_of_interest[1] + factor * (vector[int(val)][1] - point_of_interest[1]),
                           line=dict(
                               color=marker_colors[0],
                               width=1.5,
                               dash='dashdot'))
    fig_left.update_layout(coloraxis_showscale=False, title_text='<i><b>{}</b></i>'.format(title), title_font_size=12,
                           title_x=.3, title_y=.85, title_font_color='white', margin=dict(l=80, r=20, t=15, b=10))

    # Add z-coordinates in the right lower corner
    fig_left.add_annotation(dict(x=np.percentile(xy_box[0, :], 85), y=np.percentile(xy_box[1, :], 8),
                                 showarrow=False, text='<i><b>z={:0.1f}</b></i>'.format(point_of_interest[-1]),
                                 xanchor='center', font=dict(size=12, color='black')))
    fig_left.update_xaxes(showticklabels=True)
    fig_left.update_yaxes(showticklabels=True)

    return fig_left


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


def plot_similarity(rollangles, roll_values, sum_intensity, level, id, width=width_subplot, height=height_subplot):
    """plots the similarities ??"""

    # ====================    get data into a frame to facilitate plotting   ====================
    keys_to_extract = ['marker', level]
    roll_values = {key: roll_values[key] for key in keys_to_extract}
    idx_roll = [np.searchsorted(rollangles, v) for k, v in roll_values.items()]

    data2plot = pds.DataFrame([sum_intensity], index=["intensities"], columns=np.rad2deg(rollangles)).T

    data2plot['id'] = data2plot.index
    data2plot = pds.melt(data2plot, id_vars='id', value_vars=data2plot.columns[:-1])

    # ================    line plot of intensities (and estimated FFT in the case of 'marker')    ================
    fig_right = px.line(data2plot, x='id', y='value', width=width, height=height, color='variable',
                        color_discrete_map={"intensities": line_colors[0]})

    fig_right.add_hline(y=data2plot.min()['value'] * 1.05, line_width=1.2)  # somehow bulky hack for x/y-axis
    fig_right.add_vline(x=0, line_width=1.2)
    fig_right.update_traces(line=dict(width=0.5), showlegend=False)

    fig_right.update_yaxes(showticklabels=False, showgrid=False, title_text="", zerolinewidth=1.5, zeroline=True)
    fig_right.update_xaxes(showticklabels=True, showgrid=True, tickvals=np.arange(0, 61, step=20),
                           title_text="", ticks='inside', gridwidth=.25, gridcolor='lightgray')

    # ====================    display WHAT IS THE SIMILARITY INDEX?!   ====================
    for idx, rolls in enumerate(roll_values):
        fig_right.add_trace(go.Scatter(x=[np.rad2deg(rollangles[idx_roll[idx]])], y=[sum_intensity[idx_roll[idx]]],
                                       mode='markers', showlegend=False, marker=dict(size=10, color=marker_colors[0],
                                                                                     symbol='cross')))

    fig_right.update_layout(title_text="Similarity index", title_x=.5, title_y=.92,
                            margin=dict(l=20, r=20, t=45, b=10), paper_bgcolor='white', plot_bgcolor='white',
                            width=width, height=height)

    return fig_right # dcc.Graph(id=id, figure=fig_right)


def add_arrow_buttons(level):
    """adds two buttons intended to drive the display of the intensities on the left side in both z-directions"""

    return dbc.ButtonGroup([dbc.Button('\u25B2', id='{}-cranial'.format(level), size='sm'),
                            dbc.Button('\u25BC', id='{}-ventral'.format(level), size='sm')])


def vector_intensity(center, pixdim=[.7] * 3, radius=16):
    """ determines vector for a given center according to functions derived from:
    https://github.com/netstim/leaddbs/blob/master/ea_orient_intensityprofile.m"""
    from math import pi, sin, cos

    center = center[:2] if len(center) > 2 else center  # only two dimensions needed as plotted as 2D imshow
    vector = np.multiply([0, 1], radius / pixdim[0])  # use pixel dimension in the first axis ('lateral')
    vector_updated = []
    for k in range(1, 361):
        theta = (2 * pi / 360) * (k - 1)
        rotation_matrix = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
        vector_updated.append(vector @ rotation_matrix + center)

    return vector_updated


# -------------------------          Data visualisation   ------------------------- #
layout_container = html.Div(id='rotation-container', children=[
    html.H1('Please select a subject and a hemisphere to display', id='headline-tab-rotation',
            style={"margin-upper": "15px", "textAlign": "center"}),
    # First row with markers
    dbc.Row(
        [
            dcc.Loading(
                id="loading-marker-axial",
                type="default",
                children=dbc.Col(dcc.Graph(id='marker-axial'), width=4)),
            dcc.Loading(
                id="loading-marker-intensities",
                type="default",
                children=dbc.Col(dcc.Graph(id='marker-intensities'), width=4)),
            dbc.Row(children=[html.Div(id='my-knob'),
                              html.Div(id='knob-output', style={'text-align': 'center'})])
        ]),

    # Second row for Level1
    dbc.Row(
        [
            dbc.Row(children=[dbc.Col(html.Div('z-axis (Marker)', style={'text-align': 'left'})),
                              add_arrow_buttons('marker')]),
            dbc.Col(width=4),
            dbc.Col(width=4)
        ], style={"margin-left": "15rem"}),
    html.Hr(),
    dbc.Row(
        [
            dcc.Loading(
                id="loading-level1-axial",
                type="default",
                children=dbc.Col(dcc.Graph(id='level1-axial'), width=4)),
            dcc.Loading(
                id="loading-level1-intensities",
                type="default",
                children=dbc.Col(dcc.Graph(id='level1-intensities'), width=4)),
            dcc.Loading(
                id="loading-level1-similarity",
                type="default",
                children=dbc.Col(dcc.Graph(id='level1-similarity'), width=4)),
        ]),

    # Third row for Level2
    dbc.Row(
        [
            dbc.Row(children=[dbc.Col(html.Div('z-axis (Level1)', style={'text-align': 'left'})),
                              add_arrow_buttons('level1')]),
            dbc.Col(width=4),
            dbc.Col(width=4)
        ], style={"margin-left": "15rem"}),
    html.Hr(),
    dbc.Row(
        [
            dcc.Loading(
                id="loading-level2-axial",
                type="default",
                children=dbc.Col(dcc.Graph(id='level2-axial'), width=4)),
            dcc.Loading(
                id="loading-level2-intensities",
                type="default",
                children=dbc.Col(dcc.Graph(id='level2-intensities'), width=4)),
            dcc.Loading(
                id="loading-level2-similarity",
                type="default",
                children=dbc.Col(dcc.Graph(id='level2-similarity'), width=4)),
        ]),
    dbc.Row([
        dbc.Row(children=[dbc.Col(html.Div('z-axis (Level2)', style={'text-align': 'left'})),
                          add_arrow_buttons('level2')]),
        dbc.Col(width=4),
        dbc.Col(width=4)], style={"margin-left": "15rem"}),
    html.Div(offset, id='intermediate-value', style={'display': 'none'}),  # to store offset as this may be modified
    html.Div(offset, id='loaded_data', style={'display': 'none'})  # to store offset as this may be modified
])


#  =================== Callback returning headline for tab ===================
@app.callback(dash.dependencies.Output('headline-tab-rotation', 'children'),
              dash.dependencies.Input('hemisphere-headline', 'children'))
def headline_changes(value):
    """changes headline according to pressed button on sidebar"""
    if value is None:
        term = 'Please select Subject/Hemisphere'
    else:
        term = 'Rotation: {}'.format(value)

    return term


@app.callback(
    dash.dependencies.Output('knob-output', 'children'),
    [dash.dependencies.Input('my-knob', 'value')])
def update_output(value):
    if value is None:
        PreventUpdate
    else:
        return 'Rotation angle is {:.2f}.'.format(value)


@app.callback(dash.dependencies.Output('intermediate-value', 'children'),
              dash.dependencies.Input('marker-cranial', 'n_clicks'),
              dash.dependencies.Input('marker-ventral', 'n_clicks'),
              dash.dependencies.Input('level1-cranial', 'n_clicks'),
              dash.dependencies.Input('level1-ventral', 'n_clicks'),
              dash.dependencies.Input('level2-cranial', 'n_clicks'),
              dash.dependencies.Input('level2-ventral', 'n_clicks'),
              dash.dependencies.State('intermediate-value', 'children'))
def run_script_onClick(mrk_up, mrk_down, l1_up, l1_down, l2_up, l2_down, offset_value):
    if all(x is None for x in [mrk_up, mrk_down, l1_up, l1_down, l2_up, l2_down]):
        raise PreventUpdate

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'marker-cranial' in changed_id:
        vec = [1, 0, 0]
    elif 'marker-ventral' in changed_id:
        vec = [-1, 0, 0]
    elif 'level1-cranial' in changed_id:
        vec = [0, 1, 0]
    elif 'level1-ventral' in changed_id:
        vec = [0, -1, 0]
    elif 'level2-cranial' in changed_id:
        vec = [0, 0, 1]
    elif 'level2-ventral' in changed_id:
        vec = [0, 0, -1]

    new_offset = list(map(add, offset_value, vec))
    print('old: {} -> new: {}'.format(offset_value, new_offset))
    return new_offset


#  =================== Callback returning axial slices for marker, level1 and level2 ===================
@app.callback(dash.dependencies.Output('marker-axial', 'figure'),
              dash.dependencies.Output('level1-axial', 'figure'),
              dash.dependencies.Output('level2-axial', 'figure'),
              dash.dependencies.Output('my-knob', 'children'),
              dash.dependencies.Input('results-hemisphere', 'data'))
def plot_marker(value):
    """creates graphical objects for marker and level1/2 and loads rotation angle to return all for plotting purposes"""
    if value is None:
        marker_slice, level1_slice, level2_slice = [dict() for _ in range(3)]
        rotation_angle = {}
    else:
        marker_slice = prepare_imshow(np.array(value['slices']['marker']),
                                      np.array(value['valleys']['marker']),
                                      np.array(value['plot_box']['marker']),
                                      np.array(value['coordinates']['marker']),
                                      title='Marker')
        level1_slice = prepare_imshow(np.array(value['slices']['level1']),
                                      np.array(value['valleys']['level1']),
                                      np.array(value['plot_box']['level1']),
                                      np.array(value['coordinates']['level1']),
                                      title='Level1')
        level2_slice = prepare_imshow(np.array(value['slices']['level2']),
                                      np.array(value['valleys']['level2']),
                                      np.array(value['plot_box']['level2']),
                                      np.array(value['coordinates']['level2']),
                                      title='Level2')
        rotation_angle = daq.Knob(id='my-knob', size=75, value=value['angle'], max=120, min=-120, label='R/L',
                                  scale={'start': -120, 'labelInterval': 15, 'interval': 120})

    return marker_slice, level1_slice, level2_slice, rotation_angle


#  =================== Callback returning intensities for marker, level1 and level2 ===================
@app.callback(dash.dependencies.Output('marker-intensities', 'figure'),
              dash.dependencies.Output('level1-intensities', 'figure'),
              dash.dependencies.Output('level2-intensities', 'figure'),
              dash.dependencies.Input('results-hemisphere', 'data'))
def intensity_plotting(value):
    """creates figures for intensities of marker- and level1/2-artefactsand for plotting purposes"""
    level_intensities = {k: dict() for k in ['level1', 'level2']}
    if value is None:
        marker_intensity = {}
    else:
        marker_intensity = plot_intensities(np.array(value['intensities']['marker']),
                                            np.array(value['valleys']['marker']),
                                            title_id='intensity-profile-marker',
                                            fft_data=np.array(value['markerfft']),
                                            peaks=np.array(value['peak']))
        for level in level_intensities.keys():
            level_intensities[level] = plot_intensities(np.array(value['intensities'][level]),
                                                        np.array(value['valleys'][level]),
                                                        title_id='intensity-profile-{}'.format(level))

    print('these are the items in rotation: {}'.format(value.keys()))
    return marker_intensity, level_intensities['level1'], level_intensities['level2']


#  =================== Callback returning intensities for marker, level1 and level2 ===================
@app.callback(dash.dependencies.Output('level1-similarity', 'figure'),
              dash.dependencies.Output('level2-similarity', 'figure'),
              dash.dependencies.Input('results-hemisphere', 'data'))
def similarity_plotting(value):
    """creates graphical objects for similarities at level1/2 for plotting purposes"""
    if value is None:
        level1_similarity, level2_similarity = [dict() for _ in range(2)]
    else:
        level1_similarity = plot_similarity(value['roll_angles'],
                                            value['roll'],
                                            value['sum_intensities']['level1'],
                                            level='level1',
                                            id='similarity-profile-level1')
        level2_similarity = plot_similarity(value['roll_angles'],
                                            value['roll'],
                                            value['sum_intensities']['level2'],
                                            level='level2',
                                            id='similarity-profile-level2')

    return level1_similarity, level2_similarity
