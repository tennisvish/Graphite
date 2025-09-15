import dash
from dash import dcc, html, Input, Output, callback, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import base64
import io
import math
import numpy as np

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
server = app.server

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Graphite", className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='experiment-type',
                options=[
                    {'label': 'Ground Radiation', 'value': 'radiation'},
                    {'label': 'Space Flight', 'value': 'flight'},
                    {'label': 'Hypergravity', 'value': 'hypergravity'},
                    {'label': 'Combined Experiment', 'value': 'combined'}
                ],
                value='radiation',
                clearable=False
            )
        ], width=12)
    ], className="mb-4"),
    
    html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Drag and Drop or ', html.A('Select Data File')]),
                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                        'textAlign': 'center', 'margin': '10px 0'
                    }
                ),
            ], width=12),
            dbc.Col([
                html.Label("Select Payload:", className="font-weight-bold"),
                dcc.Dropdown(id='payload-dropdown', options=[], placeholder="Select payload...")
            ], width=6),
            dbc.Col([
                html.Label("Select Measurement:", className="font-weight-bold"),
                dcc.Dropdown(id='parameter-dropdown', options=[], placeholder="Select parameter...")
            ], width=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Label("Select X-Axis:", className="font-weight-bold"),
                dcc.Dropdown(
                    id='xaxis-selector',
                    options=[
                        {'label': 'Time Until Sacrifice (weeks)', 'value': 'time'},
                        {'label': 'Number of Animals', 'value': 'count'},
                        {'label': 'Total Absorbed Dose', 'value': 'dose'}
                    ],
                    value='time',
                    clearable=False
                )
            ], width=6)
        ], className="mb-3"),
        
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Compare Groups:", className="font-weight-bold"),
                    dcc.Checklist(
                        id='group-selector',
                        options=[
                            {'label': ' Show Ground Radiation', 'value': 'ground'},
                            {'label': ' Show Space Flight', 'value': 'space'},
                            {'label': ' Show Hypergravity', 'value': 'hypergravity'}
                        ],
                        value=['ground', 'space', 'hypergravity']
                    )
                ], width=12)
            ])
        ], id='combined-controls', style={'display': 'none'})
    ], id='main-controls'),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='experiment-plot',
                config={'displayModeBar': False},
                style={'height': '600px'}
            )
        ], width=12)
    ], className="mt-4"),
    
    dbc.Row([
        dbc.Col(html.Div(id='data-info', className="text-muted small"), width=12)
    ])
], fluid=True)

# Callbacks
@callback(
    [Output('main-controls', 'style'),
     Output('combined-controls', 'style')],
    Input('experiment-type', 'value')
)
def update_controls_visibility(experiment_type):
    if experiment_type == 'combined':
        return {'display': 'block'}, {'display': 'block'}
    return {'display': 'block'}, {'display': 'none'}

@callback(
    [Output('payload-dropdown', 'options'),
     Output('parameter-dropdown', 'options'),
     Output('data-info', 'children')],
    Input('upload-data', 'contents'),
    State('experiment-type', 'value'),
    prevent_initial_call=True
)
def update_dropdowns(contents, experiment_type):
    if contents is None:
        return [], [], "Upload experiment data"
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_excel(io.BytesIO(decoded))
        
        payload_options = [{'label': p, 'value': p} for p in df['Payload ID (rdrc_name)'].unique()]
        
        known_cols = ['Payload ID (rdrc_name)', 'Experimental Group (rdrc_name)',
                     'Time point of sacrifice post irradiation', 'Total absorbed dose']
        param_cols = [c for c in df.columns if c not in known_cols and pd.api.types.is_numeric_dtype(df[c])]
        param_options = [{'label': p, 'value': p} for p in param_cols]
        
        info_text = f"Loaded {len(df)} {experiment_type} experiments with {len(param_cols)} parameters"
        return payload_options, param_options, info_text
    
    except Exception as e:
        return [], [], f"Error loading file: {str(e)}"

@callback(
    Output('experiment-plot', 'figure'),
    [Input('payload-dropdown', 'value'),
     Input('parameter-dropdown', 'value'),
     Input('upload-data', 'contents'),
     Input('experiment-type', 'value'),
     Input('xaxis-selector', 'value'),
     Input('group-selector', 'value')],
    prevent_initial_call=True
)
def update_plot(selected_payload, selected_param, contents, experiment_type, xaxis_selection, group_selection):
    if experiment_type == 'radiation':
        return create_radiation_visual(selected_payload, selected_param, contents, xaxis_selection)
    elif experiment_type == 'hypergravity':
        return create_hypergravity_visual(selected_payload, selected_param, contents, xaxis_selection)
    elif experiment_type == 'flight':
        return create_flight_visual(selected_payload, selected_param, contents, xaxis_selection)
    elif experiment_type == 'combined':
        return create_combined_visual(contents, group_selection)
    return px.scatter(title="Select an experiment type")

def create_radiation_visual(selected_payload, selected_param, contents, xaxis_selection):
    if not all([selected_payload, contents]):
        return px.scatter(title="Select a payload and upload data")
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_excel(io.BytesIO(decoded))
        plot_df = df[df['Payload ID (rdrc_name)'] == selected_payload].copy()
        
        # Create custom sorting key to order doses: Sham first, then ascending
        def dose_sort_key(x):
            if x == 0:  # Sham
                return (0, 0)
            else:
                return (1, x)  # Then sort by dose value
        
        # Sort by our custom key (ascending=True puts Sham first, then doses in order)
        plot_df = plot_df.sort_values(
            'Total absorbed dose',
            key=lambda x: x.map(dose_sort_key),
            ascending=True
        )
        
        # Create group labels with counts and wrap text after 20 characters
        group_counts = plot_df['Experimental Group (rdrc_name)'].value_counts().to_dict()
        plot_df['Group_Count'] = plot_df['Experimental Group (rdrc_name)'].map(group_counts)
        
        def wrap_text(text, count, max_length=20):
            wrapped = []
            current_line = ""
            words = text.split()
            for word in words:
                if len(current_line) + len(word) + 1 <= max_length:
                    current_line += (" " + word if current_line else word)
                else:
                    wrapped.append(current_line)
                    current_line = word
            if current_line:
                wrapped.append(current_line)
            wrapped_text = "<br>".join(wrapped)
            return f"{wrapped_text} ({count})"
        
        plot_df['Y_Label'] = plot_df.apply(
            lambda row: wrap_text(row['Experimental Group (rdrc_name)'], row['Group_Count']),
            axis=1
        )
        
        # Maintain this order in the visualization
        ordered_groups = plot_df['Y_Label'].unique()
        
        # Determine x-axis column and title
        if xaxis_selection == 'time':
            x_col = 'Time point of sacrifice post irradiation'
            x_title = 'Time Until Sacrifice (weeks)'
            x_range = [0, math.ceil(plot_df[x_col].max() * 1.05)]
        elif xaxis_selection == 'count':
            x_col = 'Group_Count'
            x_title = 'Number of Animals'
            x_range = [0, math.ceil(plot_df[x_col].max() * 1.1)]
        else:  # dose
            x_col = 'Total absorbed dose'
            x_title = 'Total Absorbed Dose (cGy)'
            x_range = [0, math.ceil(plot_df[x_col].max() * 1.1)]
        
        fig = px.bar(
            plot_df,
            x=x_col,
            y='Y_Label',
            color='Total absorbed dose',
            color_continuous_scale='thermal',
            orientation='h',
            hover_data={
                'Time point of sacrifice post irradiation': False,
                'Total absorbed dose': ':.1f',
                'Y_Label': False,
                'Group_Count': True
            },
            height=600,
            range_x=x_range,
            category_orders={'Y_Label': ordered_groups}
        )
        
        if selected_param:
            fig.update_traces(
                hovertext=plot_df[selected_param].apply(lambda x: f"{selected_param}: {x:.2f}"),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"{x_title}: %{{x:.0f}}<br>"
                    "Dose: %{marker.color:.1f} cGy<br>"
                    "%{hovertext}<br>"
                    "N = %{customdata[0]}<extra></extra>"
                ),
                customdata=plot_df[['Group_Count']]
            )
        
        # Add radiation icons inside bars
        if xaxis_selection != 'dose':
            irradiated_mask = plot_df['Total absorbed dose'] > 0
            irradiated_groups = plot_df[irradiated_mask]
            
            if not irradiated_groups.empty:
                max_dose = irradiated_groups['Total absorbed dose'].max()
                min_size, max_size = 12, 24
                
                group_data = plot_df[['Y_Label', 'Total absorbed dose']].drop_duplicates()
                
                for _, row in group_data.iterrows():
                    if row['Total absorbed dose'] > 0:
                        dose = row['Total absorbed dose']
                        icon_size = min_size + (max_size - min_size) * (dose / max_dose)
                        
                        fig.add_annotation(
                            x=1,
                            y=row['Y_Label'],
                            text="☢️",
                            showarrow=False,
                            xref="x",
                            yref="y",
                            xanchor="left",
                            yanchor="middle",
                            font=dict(size=icon_size),
                            xshift=10,
                            bgcolor="rgba(255,255,255,0.7)",
                            bordercolor="rgba(0,0,0,0.1)",
                            borderpad=2
                        )
        
        # Add metadata for Chang/Blakely payloads
        if "Chang" in selected_payload or "Blakely" in selected_payload:
            fig.add_annotation(
                x=0.5,
                y=1.1,
                xref="paper",
                yref="paper",
                text="<b>Metadata:</b> Irradiation Facility: NSRL | Mice: Female CB6F1",
                showarrow=False,
                font=dict(size=12, color="#666"),
                align="center",
                bordercolor="#ccc",
                borderwidth=1,
                borderpad=4,
                bgcolor="rgba(255,255,255,0.8)"
            )
        
        fig.update_layout(
            plot_bgcolor='rgba(250,250,250,1)',
            paper_bgcolor='rgba(250,250,250,1)',
            font={'family': "SF Pro Display, -apple-system, sans-serif", 'color': '#333'},
            title={'text': f"<b>{selected_payload}</b><br><span style='font-size:0.8em;color:#666'>Radiation Experiment</span>", 'y':0.95},
            margin={'l': 120, 'r': 40, 't': 150 if ("Chang" in selected_payload or "Blakely" in selected_payload) else 100, 'b': 60},
            xaxis={
                'title': x_title,
                'range': x_range,
                'gridcolor': 'rgba(230,230,230,1)'
            },
            yaxis={
                'title': '',
                'autorange': 'reversed',
                'tickmode': 'array',
                'tickvals': plot_df['Y_Label'].unique()
            },
            coloraxis_colorbar={
                'title': 'Dose (cGy)',
                'thickness': 10,
                'len': 0.5,
                'y': 0.5
            },
            hoverlabel={
                'bgcolor': 'white',
                'font_size': 12,
                'font_family': "SF Pro Display"
            },
            bargap=0.2
        )
        
        fig.update_traces(
            width=0.6,
            marker_line_color='white',
            marker_line_width=0.5,
            opacity=0.9,
            base=0
        )
        
        return fig
    
    except Exception as e:
        return px.scatter(title=f"Error: {str(e)}")
        
def create_hypergravity_visual(selected_payload, selected_param, contents, xaxis_selection):
    if not all([selected_payload, contents]):
        return px.scatter(title="Select a payload and upload data")
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_excel(io.BytesIO(decoded))
        plot_df = df[df['Payload ID (rdrc_name)'] == selected_payload].copy()
        
        # Same custom sorting key
        def dose_sort_key(x):
            if x == 0:  # Sham
                return (0, 0)
            else:
                return (1, x)
        
        # Sort by dose (ascending=True puts Sham first, then doses in order)
        plot_df = plot_df.sort_values(
            'Total absorbed dose',
            key=lambda x: x.map(dose_sort_key),
            ascending=True
        )
        
        # Create combined group labels with animal counts
        plot_df['Combined_Group'] = plot_df['Experimental Group (rdrc_name)'].astype(str) + ' + ' + \
                                   plot_df['Total absorbed dose'].astype(str) + ' cGy'
        group_counts = plot_df['Combined_Group'].value_counts().to_dict()
        plot_df['Group_Count'] = plot_df['Combined_Group'].map(group_counts)
        plot_df['Y_Label'] = plot_df['Combined_Group'] + ' (' + plot_df['Group_Count'].astype(str) + ')'
        
        # Maintain this order in the visualization
        ordered_groups = plot_df['Y_Label'].unique()
        
        # Determine x-axis column and title
        if xaxis_selection == 'time':
            x_col = 'Time point of sacrifice post irradiation'
            x_title = 'Time Until Sacrifice (weeks)'
            x_range = [0, math.ceil(plot_df[x_col].max() * 1.05)]
        elif xaxis_selection == 'count':
            x_col = 'Group_Count'
            x_title = 'Number of Animals'
            x_range = [0, math.ceil(plot_df[x_col].max() * 1.1)]
        else:  # dose
            x_col = 'Total absorbed dose'
            x_title = 'Total Absorbed Dose (cGy)'
            x_range = [0, math.ceil(plot_df[x_col].max() * 1.1)]
        
        fig = px.bar(
            plot_df,
            x=x_col,
            y='Y_Label',
            color='Total absorbed dose',
            color_continuous_scale='thermal',
            orientation='h',
            hover_data={
                'Time point of sacrifice post irradiation': False,
                'Total absorbed dose': ':.1f',
                'Y_Label': False,
                'Group_Count': True
            },
            height=600,
            range_x=x_range,
            category_orders={'Y_Label': ordered_groups}
        )
        
        if selected_param:
            fig.update_traces(
                hovertext=plot_df[selected_param].apply(lambda x: f"{selected_param}: {x:.2f}"),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"{x_title}: %{{x:.0f}}<br>"
                    "Dose: %{marker.color:.1f} cGy<br>"
                    "%{hovertext}<br>"
                    "N = %{customdata[0]}<extra></extra>"
                ),
                customdata=plot_df[['Group_Count']]
            )
        
        fig.update_layout(
            plot_bgcolor='rgba(250,250,250,1)',
            paper_bgcolor='rgba(250,250,250,1)',
            font={'family': "SF Pro Display, -apple-system, sans-serif", 'color': '#333'},
            title={'text': f"<b>{selected_payload}</b><br><span style='font-size:0.8em;color:#666'>Hypergravity Experiment</span>"},
            margin={'l': 150, 'r': 40, 't': 100, 'b': 60},
            xaxis={
                'title': x_title,
                'range': x_range,
                'gridcolor': 'rgba(230,230,230,1)'
            },
            yaxis={
                'title': '',
                'autorange': 'reversed',
                'tickmode': 'array',
                'tickvals': plot_df['Y_Label'].unique()
            },
            coloraxis_colorbar={
                'title': 'Dose (cGy)',
                'thickness': 10,
                'len': 0.5,
                'y': 0.5
            },
            hoverlabel={
                'bgcolor': 'white',
                'font_size': 12,
                'font_family': "SF Pro Display"
            },
            bargap=0.2
        )
        
        fig.update_traces(
            width=0.6,
            marker_line_color='white',
            marker_line_width=0.5,
            opacity=0.9,
            base=0
        )
        
        return fig
    
    except Exception as e:
        return px.scatter(title=f"Error: {str(e)}")


def create_flight_visual(selected_payload, selected_param, contents, xaxis_selection):
    if not all([selected_payload, contents]):
        return px.scatter(title="Select a payload and upload data")
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_excel(io.BytesIO(decoded))
        plot_df = df[df['Payload ID (rdrc_name)'] == selected_payload].copy()
        
        # Create group labels with animal counts
        group_counts = plot_df['Experimental Group (rdrc_name)'].value_counts().to_dict()
        plot_df['Group_Count'] = plot_df['Experimental Group (rdrc_name)'].map(group_counts)
        plot_df['Y_Label'] = plot_df['Experimental Group (rdrc_name)'] + ' (' + plot_df['Group_Count'].astype(str) + ')'
        
        # Determine x-axis column and title
        if xaxis_selection == 'time':
            x_col = 'Time point of sacrifice post irradiation'
            x_title = 'Time Until Sacrifice (weeks)'
            x_range = [0, math.ceil(plot_df[x_col].max() * 1.05)]
        elif xaxis_selection == 'count':
            x_col = 'Group_Count'
            x_title = 'Number of Animals'
            x_range = [0, math.ceil(plot_df[x_col].max() * 1.1)]
        else:  # dose
            x_col = 'Total absorbed dose'
            x_title = 'Total Absorbed Dose (cGy)'
            x_range = [0, math.ceil(plot_df[x_col].max() * 1.1)]
        
        # Create the visualization
        fig = px.bar(
            plot_df,
            x=x_col,
            y='Y_Label',
            color='Gravity level',  # Using Gravity level for color
            orientation='h',
            hover_data={
                'Time point of sacrifice post irradiation': False,
                'Gravity level': True,
                'Y_Label': False,
                'Group_Count': True
            },
            height=600,
            range_x=x_range,
            category_orders={'Y_Label': plot_df['Y_Label'].unique()}
        )
        
        if selected_param:
            fig.update_traces(
                hovertext=plot_df[selected_param].apply(lambda x: f"{selected_param}: {x:.2f}"),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"{x_title}: %{{x:.0f}}<br>"
                    "Gravity: %{marker.color}<br>"
                    "%{hovertext}<br>"
                    "N = %{customdata[0]}<extra></extra>"
                ),
                customdata=plot_df[['Group_Count']]
            )
        
        fig.update_layout(
            plot_bgcolor='rgba(250,250,250,1)',
            paper_bgcolor='rgba(250,250,250,1)',
            font={'family': "SF Pro Display, -apple-system, sans-serif", 'color': '#333'},
            title={'text': f"<b>{selected_payload}</b><br><span style='font-size:0.8em;color:#666'>Space Flight Experiment</span>"},
            margin={'l': 150, 'r': 40, 't': 100, 'b': 60},
            xaxis={
                'title': x_title,
                'range': x_range,
                'gridcolor': 'rgba(230,230,230,1)'
            },
            yaxis={
                'title': '',
                'autorange': 'reversed',
                'tickmode': 'array',
                'tickvals': plot_df['Y_Label'].unique()
            },
            hoverlabel={
                'bgcolor': 'white',
                'font_size': 12,
                'font_family': "SF Pro Display"
            },
            bargap=0.2,
            legend_title="Gravity Level"
        )
        
        fig.update_traces(
            width=0.6,
            marker_line_color='white',
            marker_line_width=0.5,
            opacity=0.9,
            base=0
        )
        
        return fig
    
    except Exception as e:
        return px.scatter(title=f"Error: {str(e)}")

def create_combined_visual(contents, group_selection):
    return px.scatter(title="Combined experiment visualization coming soon")

if __name__ == '__main__':
    app.run(debug=True)
