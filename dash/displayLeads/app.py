import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dtc
from dash.dependencies import Input, Output, State, ClientsideFunction
import pickle
import dash_bootstrap_components as dbc

from time import sleep
from progress.spinner import MoonSpinner

import sys
import time
import os
import numpy as np
import pandas as pds
from ManualCorrectionDash import DashDataReturn
import plotly.graph_objects as go

from datetime import datetime as dt
import pathlib

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Get data for the contents of the GUI
print("Extracting all necessary information ...", flush=True)
inputfolder = '/media/storage/cDBS/data/NIFTI/subj1/' # TODO: this must be included somehow somewhere
subject = 'subj1'
filename_leadmodel = os.path.join(inputfolder, 'elecModels_' + subject + '.pkl')
lead_data, _, _ = DashDataReturn.load_leadModel(inputfolder, filename=filename_leadmodel)
lead_model, default_positions, default_coordinates = DashDataReturn.get_default_lead(lead_data[0])  # all in [mm]
lead_data, sides = DashDataReturn.estimate_hemisphere(lead_data)

marker, coordinates, trajectory, resize, emp_dist = [{} for _ in range(5)]
sides = ['left', 'right']
for hemisphere in sides:
    marker[hemisphere], coordinates[hemisphere], trajectory[hemisphere], resize[hemisphere] = \
                DashDataReturn.get_leadModel(lead_data[hemisphere], default_positions, default_coordinates, side=hemisphere)
intensity_matrix, bounding_box, fitvolume = DashDataReturn.get_volumetry(lead_data, trajectory)
print("Done", end ='')

server = app.server
app.config.suppress_callback_exceptions = True


def plotCTintensitiesPlotLy(intensity_matrix, fitvolume, data_surface):
    """function plotting perpendicular images of the intensities obtained from trajectory coordinates; separate
    function needed here as changes occur in trejectories/markers"""

    for idx, (direction, intensity) in enumerate(intensity_matrix.items()):
        matrix2plot = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))  # Prepare data!
        X, Y, Z = fitvolume[direction][0, :, :], fitvolume[direction][1, :, :], fitvolume[direction][2, :, :]
        slice_temp = get_the_slice(X, Y, Z, matrix2plot)
        data_surface.append(slice_temp)

    return data_surface

def get_the_slice(x, y, z, surfacecolor, colorscale='Greys', showscale=False):
    return go.Surface(x=x,  # https://plot.ly/python/reference/#surface
                      y=y,
                      z=z,
                      surfacecolor=surfacecolor,
                      colorscale=colorscale,
                      showscale=showscale,
                      name="")

def get_coordinates(data, coordinates):
    """add the coordinates to the surface plots generated so far"""
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
        data.append(trace)

    return data

data_surface = []
data_surface = get_coordinates(data_surface, coordinates[hemisphere])
data_surface = plotCTintensitiesPlotLy(intensity_matrix[hemisphere], fitvolume[hemisphere], data_surface)


def estimate_hemisphere(lead_data):
        """estimates the available sides and returns a list of all available leads; as all data is in LPS (Left,
        Posterior, Superior) so that side can be deduced from trajectory"""

        sides = []
        renamed_lead_data = dict()
        for info in lead_data:
            side_temp = 'right' if not info['trajectory'][0, 0] > info['trajectory'][-1, 0] else 'left'
            renamed_lead_data[side_temp] = info
            sides.append(side_temp)
        return renamed_lead_data, sides


def create_dataframe(subj, lead_data, side, marker_prefix=''):
    """summarises all results from the lead data in a pandas Dataframe as input for the table in information"""

    marker_key = '' if marker_prefix =='' else " ({}) ".format(marker_prefix)

    general_information, marker_information = dict(), dict()
    general_information['Subj'] = subj
    general_information['Model'] = lead_data['model']
    general_information['Side'] = '{} hemisphere'.format(side)
    general_information['Rotation'] = lead_data['rotation']
    general_information['Filename CT'] = lead_data['filenameCTimaging'][1]

    marker_default = {'Marker Head{}': 'markers_head{}',
    'Marker 1{}': 'markers_x{}',
    'Marker 2{}': 'markers_y{}',
    'Marker tail{}': 'markers_tail{}'}


    for k, v in marker_default.items():
        try:
            lead_data[v.format(marker_prefix)] = np.array([item for sublist in lead_data[v.format(marker_prefix)] for item in sublist])
        except TypeError:
            pass

        try:
            marker_information[k.format(marker_key)] = tuple(lead_data[v.format(marker_prefix)]) if \
                type(lead_data[v.format(marker_prefix)]) == np.ndarray else \
                lead_data[v.format(marker_prefix)]
        except KeyError:
            marker_information[k.format(marker_key)] = tuple(lead_data[v.format('')]) if \
                type(lead_data[v.format('')]) == np.ndarray else lead_data[v.format('')]


    return pds.DataFrame.from_dict(data=general_information, orient='index'), \
           pds.DataFrame.from_dict(data=marker_information, orient='index', columns=['x', 'y', 'z'])

information, markers = create_dataframe('subj1', lead_data['left'], 'left', marker_prefix='')


def title():
    return html.Div(
        id="title-entire",
        children=[
            html.H1("Display Lead Information"),
        ],
        style={"margin-left": "1%"}
    )

def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.Br(),
            html.H2("At this point, lead information such as coordinates or rotation can be checked and modified"),
            html.Div(
                id="intro",
                children="Explore lead information and modify the obtained results. Click on the different options to start visualising the information.",
            ),
        ],
    )

def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    list_of_leads = ['leftSTN', 'rightSTN']
    return html.Div(
        id="control-card",
        children=[
            html.Br(),
            html.H3("Lead to modify:", style={"margin-upper": "15px"}),
            dcc.Dropdown(
                id="clinic-select",
                options=[{"label": i, "value": i} for i in list_of_leads],
                value=list_of_leads[0],
            ),
        ]
    )

app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("plotly_logo.png"),  style={'height':'5%', 'width':'5%',
                                                                                 'padding': 15, "margin-left": "0.5%"})],
        ),
        html.Div(
            id="title",
            className="title-entirely",
            children=[title()]
        ),
        html.Br(),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),

        # Right columns
        html.Div([
            dcc.Tabs(id='tabs', value='tab-visualise', style={'width': '70%'},
                children=[
                dcc.Tab(label='Visualisation',
                        value='tab-visualise',
                        className='custom-tab',
                        selected_className='custom-tab--selected'),
                dcc.Tab(label='Modify Markers',
                        value='tab-markers',
                        className='custom-tab',
                        selected_className='custom-tab--selected'),
                dcc.Tab(label='View Rotation',
                        value='tab-rotation',
                        className='custom-tab',
                        selected_className='custom-tab--selected'),
                dcc.Tab(label='Lead information',
                        value='tab-information',
                        className='custom-tab',
                        selected_className='custom-tab--selected'),
            ]),
            html.Div(id='tabs-content', style={'height': '40%', 'width': '45%', 'float': 'left'})
        ])
    ],
)

# Callbacks


@app.callback(
    Output("tabs-content", "children"),
    [Input("tabs", "value")])
def render_tab_content(value):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """

    def conditional_formatter(value):
        return "{:.2f}".format(value) if not isinstance(value, str) else value

    def generate_information_summary(key, value):
        value = '0*' if value == [] else value
        return html.H6(["{}: {}".format(key, value)])

    if value == "tab-information":
        return html.Div(children=[
            html.Br(),
            html.H5("General information:", style={"margin-upper": "15px"}),
            dbc.Col(children=[generate_information_summary(k,v) for k,v in information.to_dict()[0].items()]),
            html.Br(),
            html.H5("Saved markers:", style={"margin-upper": "15px"}),
            dtc.DataTable(id='markerTable',
                          columns=[{"name": i, "id": i} for i in markers.reset_index().columns],
                          data=markers.applymap(conditional_formatter).reset_index().to_dict('rows'),
                          editable=False,
                          #style_cell={'width': '{}%'.format(len(markers.columns))})
                          ),
            dcc.RadioItems(id='input-radio-button',
                           options=[dict(label='Saved', value='saved'),
                                    dict(label='Default', value='default')],
                           value='saved')
        ]
        )
    elif value == "tab-visualise":

        layout = go.Layout(
            width=600,
            height=500,
            autosize=False,
            margin=go.layout.Margin(
                l=5,
                r=5,
                b=5,
                t=5,
                pad=0
            ),
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

        return html.Div(children=[
            html.Br(),
            html.H5("Surface plot: Left hemisphere", style={"margin-upper": "15px"}),
            dcc.Graph(id='surfacePlot',
                      figure={'data': data_surface,
                              'layout': layout}),
            ],
        )
    else:
        return html.Div([html.H2("Tab content 2")])


# Radio -> multi
@app.callback(Output('markers', 'value'),
              [Input('input-radio-button', 'value')])
def display_type(selector):
    if selector == 'saved':
        _, markers = markers = create_dataframe('subj1', lead_data['left'], 'left', marker_prefix='')
    elif selector == 'default':
        _, markers = markers = create_dataframe('subj1', lead_data['left'], 'left', marker_prefix='def')
        return ['GD', 'GE', 'GW', 'IG', 'IW', 'OD', 'OE', 'OW']
    else:
        pass

    return markers

# Run the server
if __name__ == "__main__":

    app.run_server(debug=True)
