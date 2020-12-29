import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
import pathlib
import numpy as np
import pandas as pds
from datasets.leadRotation import PrepareData
import plotly.graph_objects as go
import plotly.express as px

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

# TODO: this should be moved somewhere in a file wher it is initiaited
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

# Settings and data
# ==============================    Settings   ==============================
width_subplot = 300
height_subplot = 250

# ==============================    Data   ==============================
print("Extracting rotation ...", flush=True)
inputfolder = '/media/storage/cDBS/data/NIFTI/' # TODO: this must be included somehow somewhere
subject = 'subj3'
p = PrepareData()
# angles, slices, intensities_marker_angle, intensities_level_angle, sum_intensity, roll, dir_valleys, markerfft, \
# valleys, peaks, vector_marker, vector_level, directional_coord, marker_coord, boxes = \
rotation = p.getData(subj=subject, input_folder=inputfolder)
print("Done", end ='')


dfv = pd.read_csv(DATA_PATH.joinpath("vgsales.csv"))  # GregorySmith Kaggle
sales_list = ["North American Sales", "EU Sales", "Japan Sales", "Other Sales",	"World Sales"]


# ------------------------- Plotting part  ------------------------- #

def prepare_imshow(data, valleys, vector, xy_box, point_of_interest, title='Marker',
                   width=width_subplot, height=height_subplot, color_scale='ice', clims=(-50, 100), radius=20):
    """ this part plots the intensities of the marker and the different levels at which rotation is estimated"""

    fig_left = px.imshow(data.T,
                         x=xy_box[0,:], y=xy_box[1,:],
                         color_continuous_scale=color_scale, width=width, height=height, range_color=clims, aspect='equal', origin='lower')
    directions = dict({ 'P': [point_of_interest[0], np.percentile(xy_box[1,:], 8)],
                        'A': [point_of_interest[0], np.percentile(xy_box[1,:], 92)],
                        'R': [np.percentile(xy_box[0,:], 8), point_of_interest[1]],
                        'L': [np.percentile(xy_box[0,:], 92), point_of_interest[1]]})# 'M': [], 'L': []}
    for ann, value in directions.items():
        fig_left.add_annotation(dict(x=value[0], y=value[1],
                                     showarrow=False, text=ann, xanchor='center', font=dict(size=20, color='white')))

        #    fig_left.add_annotation(dict(x=point_of_interest[1], y=np.percentile(xy_box[:,0], 55),
        #                                 showarrow=False, text="A", xanchor='center', font=dict(size=20, color='white')))
    fig_left.add_trace(go.Scatter(x=[point_of_interest[0]], y=[point_of_interest[1]], mode='markers',
                                  marker_symbol='circle-open',
                                  marker=dict(size=10,
                                              color='#d65d0e')))


    factor = .2 if title == 'Marker' else .2
    fig_left.add_shape(type='circle',
                       xref="x", yref="y",
                       x0=point_of_interest[0] - radius / 2,
                       y0=point_of_interest[1] - radius / 2,
                       x1=point_of_interest[0] + radius / 2,
                       y1=point_of_interest[1] + radius / 2,
                       line=dict(
                           color='#d65d0e',
                           width=.75,
                           dash='dot',
                       ))
    for val in valleys:
        fig_left.add_shape(type='line',
                           x0=point_of_interest[0],
                           x1=point_of_interest[0] + factor * (vector[int(val)][0] - point_of_interest[0]),
                           y0=point_of_interest[1],
                           y1=point_of_interest[1] + factor * (vector[int(val)][1] - point_of_interest[1]),
                           line=dict(
                               color='#b16286',
                               width=1.5,
                               dash='dashdot'))
    fig_left.update_layout(coloraxis_showscale=False, title_text='<i><b>{}</b></i>'.format(title), title_font_size=12,
                           title_x=.3, title_y=.85, title_font_color='white', margin=dict(l=80, r=20, t=15, b=10))
    fig_left.update_xaxes(showticklabels=True)
    fig_left.update_yaxes(showticklabels=True)

    return dcc.Graph(id='{}-rotation'.format(title).lower(), figure=fig_left)

def plot_marker_intensities(intensities, markerfft, peaks, valleys, id, width=width_subplot, height=height_subplot):
    """ intensity of of the data at the 360 degrees around the marker"""

    angles = intensities[:,1]
    data2plot = pds.DataFrame([intensities[:,0], markerfft], index=["intensities", "fft"],
                              columns=np.rad2deg(angles)).T
    data2plot['id'] = data2plot.index
    data2plot = pds.melt(data2plot, id_vars='id', value_vars=data2plot.columns[:-1])

    fig = px.line(data2plot, x='id', y='value', width=width, height=height, color='variable',
                  color_discrete_map = {
                      "intensities": "#456987",
                      "fft": "#147852"})
    fig.update_traces(line=dict(width=0.5), showlegend=False)

    fig.update_yaxes(showticklabels=False, showgrid=False, title_text="", zerolinewidth=1.5, zeroline=True)
    fig.update_xaxes(showticklabels=True, showgrid=False, tickvals=np.arange(0, 361, step=60),
                     title_text="", ticks='inside')
    fig.add_trace(go.Scatter(x=[np.rad2deg(angles)], y=[markerfft], name='fft', mode='lines', showlegend=False))

    for p,v in zip(peaks, valleys):
        fig.add_trace(go.Scatter(x=[np.rad2deg(angles[int(p)])], y=[intensities[int(p), 0]],
                                 mode='markers', showlegend=False, marker=dict(size=10, color='#2ca02c')))
        fig.add_trace(go.Scatter(x=[np.rad2deg(angles[int(v)])], y=[intensities[int(v), 0]],
                                 mode='markers', showlegend=False, marker=dict(size=10, color='#ff7f0e')))

    fig.update_layout(title_text="Intensity profile", title_x=.5, title_y=.95,
                      margin=dict(l=20, r=20, t=45, b=10), paper_bgcolor='white', plot_bgcolor='white',
                      width=width, height=height)

    return dcc.Graph(id=id, figure=fig)

def plot_level_intensities(intensities, valleys, id, width=width_subplot, height=height_subplot):
    """ intensity of of the data at the 360 degrees around the marker"""

    angles = intensities[:,1]
    fig_middle = px.line(intensities[:,0], width=width, height=height, color='variable',
                  color_discrete_map = {
                      "intensities": "#456987",
                      "fft": "#147852"})
    fig_middle.update_traces(line=dict(width=0.5), showlegend=False)

    fig_middle.update_yaxes(showticklabels=True, showgrid=True, range=[-200, 200], title_text="", zerolinewidth=1.5, zeroline=True)
    fig_middle.update_xaxes(showticklabels=True, showgrid=False, tickvals=np.arange(0, 361, step=60),
                     title_text="", ticks='inside')
    for v in valleys:
        fig_middle.add_trace(go.Scatter(x=[np.rad2deg(angles[int(v)])], y=[intensities[int(v), 0]],
                                 mode='markers', showlegend=False, marker=dict(size=10, color='#ff7f0e')))

    fig_middle.update_layout(title_text="Intensity profile", title_x=.5, title_y=.95,
                      margin=dict(l=20, r=20, t=45, b=10), paper_bgcolor='white', plot_bgcolor='white',
                      width=width, height=height)

    return dcc.Graph(id=id, figure=fig_middle)

def plot_similarity(rollangles, roll_values, sum_intensity, level, id, width=width_subplot, height=height_subplot):
    """plots the similarities ??"""

    keys_to_extract = ['marker', level]
    roll_values = {key: roll_values[key] for key in keys_to_extract}
    idx_roll = [np.searchsorted(rollangles, v) for k,v in roll_values.items()]

    data2plot = pds.DataFrame([sum_intensity, np.rad2deg(rollangles)], index=["intensities", "angles"]).T
    fig_right = px.line(data2plot, x='angles', y='intensities', width=width, height=height)
#                         color_discrete_map={
#                             "intensities": "#456987",
#                             "fft": "#147852"})
    fig_right.update_traces(line=dict(width=0.5), showlegend=False)

    fig_right.update_yaxes(showticklabels=False, showgrid=False, title_text="", zerolinewidth=1.5, zeroline=True)
    fig_right.update_xaxes(showticklabels=True, showgrid=False, tickvals=np.arange(0, 61, step=20),
                           title_text="", ticks='inside')
    for idx, rolls in enumerate(roll_values):
        fig_right.add_trace(go.Scatter(x=[np.rad2deg(rollangles[idx_roll[idx]])], y=[sum_intensity[idx_roll[idx]]],
                                        mode='markers', showlegend=False, marker=dict(size=10, color='#ff7f0e')))

    fig_right.update_layout(title_text="Similarity index", title_x=.5, title_y=.92,
                      margin=dict(l=20, r=20, t=45, b=10), paper_bgcolor='white', plot_bgcolor='white',
                      width=width, height=height)


    return dcc.Graph(id=id, figure=fig_right)

def add_arrow_buttons(level):
    """adds two buttons intended to drive the display of the intensities on the left side in both z-directions"""

    return dbc.ButtonGroup([dbc.Button('\u25B2', id='{}-cranial'.format(level), size='sm'),
                            dbc.Button('\u25BC', id='{}-ventral'.format(level), size='sm')])


# ------------------------- Layout part  ------------------------- #


layout_container = html.Div(id='rotation-container', children=[
    html.H1("Rotation: Left hemisphere", style={"margin-upper": "15px", "textAlign": "center"}),
    dbc.Row([
        dbc.Col(prepare_imshow(rotation['slices']['marker'],
                               rotation['valleys']['marker'],
                               rotation['vector']['marker'],
                               rotation['plot_box']['marker'],
                               rotation['coordinates']['marker'],
                               title='Marker'), width=4),
        dbc.Col(plot_marker_intensities(rotation['intensities']['marker'],
                                        rotation['markerfft'],
                                        rotation['peak'],
                                        rotation['valleys']['marker'],
                                        id='intensity-profile-marker'), width=4)]),
    dbc.Row([
        dbc.Col(add_arrow_buttons('marker'), width=4),
        dbc.Col(width=4),
        dbc.Col(width=4)], style={"margin-left": "15rem"}),
    dbc.Row([
        dbc.Col(prepare_imshow(rotation['slices']['level1'],
                               rotation['valleys']['level1'],
                               rotation['vector']['level1'],
                               rotation['plot_box']['level1'],
                               rotation['coordinates']['level1'],
                               title='Level1'), width=4),
        dbc.Col(plot_level_intensities(rotation['intensities']['level1'],
                                        rotation['valleys']['level1'],
                                        id='intensity-profile-level1'), width=4),
        dbc.Col(plot_similarity(rotation['roll_angles'],
                                rotation['roll'],
                                rotation['sum_intensities']['level1'],
                                level='level1',
                                id='similarity-profile-level1'), width=4)]),
    dbc.Row([
        dbc.Col(add_arrow_buttons('level1'), width=4),
        dbc.Col(width=4),
        dbc.Col(width=4)], style={"margin-left": "15rem"}),
    dbc.Row([
        dbc.Col(prepare_imshow(rotation['slices']['level2'],
                               rotation['valleys']['level2'],
                               rotation['vector']['level2'],
                               rotation['plot_box']['level2'],
                               rotation['coordinates']['level2'],
                               title='Level2'), width=4),
        dbc.Col(plot_level_intensities(rotation['intensities']['level2'],
                                       rotation['valleys']['level2'],
                                       id='intensity-profile-level2'), width=4),
        dbc.Col(plot_similarity(rotation['roll_angles'],
                                rotation['roll'],
                                rotation['sum_intensities']['level2'],
                                level='level2',
                                id='similarity-profile-level2'), width=4)]),
    dbc.Row([
        dbc.Col(add_arrow_buttons('level2'), width=4),
        dbc.Col(width=4),
        dbc.Col(width=4)], style={"margin-left": "15rem"}),
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
