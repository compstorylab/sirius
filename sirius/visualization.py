import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import plotly
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.offline import download_plotlyjs, plot, iplot, init_notebook_mode
import json


# Discrete-Discrete Confusion Matrices
def DD_viz(df, charter='Plotly', output_chart=True, output_dir=None, resolution=150):
    ''' Takes a filtered dataframe of two discrete feature columns and generates a heatmap '''

    U = df.columns[0]
    V = df.columns[1]

    i_range = df[U].unique()
    j_range = df[V].unique()
    s = pd.DataFrame(columns=i_range, index=j_range)
    for i in i_range:
        for j in j_range:
            s[i][j] = df[(df[U] == i) & (df[V] == j)].filter([U, V], axis=1).shape[0]
            mutual_support = s.sum().sum()

    s = s.astype(int)

    if charter == 'Plotly':
        fig = ff.create_annotated_heatmap(
            s.values,
            x=[str(i) for i in i_range],
            y=[str(j) for j in j_range],
            colorscale='Blues'
        )
        fig.update_layout(
            xaxis_title=U.replace('_', ' ').title(),
            yaxis_title=V.replace('_', ' ').title(),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )

        if output_chart:
            fig.update_xaxes(tickcolor='white', tickfont=dict(color='white'))
            fig.update_yaxes(tickcolor='white', tickfont=dict(color='white'))
            fig.update_layout(font=dict(color="white"))
            fig.write_image(str(output_dir / 'charts' / (U + '_' + V + '.png')), scale=resolution // 72)
    else:
        plt.clf()
        plt.figure(dpi=resolution)
        sns.heatmap(s, annot=True, cmap="Blues", cbar=False, linewidths=1)
        plt.xlabel(U.replace('_', ' ').title())
        plt.ylabel(V.replace('_', ' ').title())
        if output_chart:
            plt.savefig(output_dir / 'charts' / (U + '_' + V + '.png'), dpi=resolution)

    plt.close('all')


# Discrete-Continuous Violin Plots
def DC_viz(df, charter='Plotly', output_chart=False, output_dir=None, resolution=150, discrete_first=True):
    ''' Takes a subset dataframe of one continuous and one discrete feature and generates a Violin Plot '''

    U = df.columns[0]
    V = df.columns[1]
    if discrete_first:
        D = U
        C = V
    else:
        D = V
        C = U

    if charter == 'Plotly':
        fig = go.Figure()
        for i in list(df[D].unique()):
            series = df[df[D] == i][C]
            fig.add_trace(go.Violin(x=series, name=str(i)))

        fig.update_traces(orientation='h', side='positive', width=3, points=False)
        fig.update_layout(
            xaxis_showgrid=False,
            xaxis_zeroline=False,
            xaxis_title=C.replace('_', ' ').title(),
            yaxis_title=D.replace('_', ' ').title(),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            showlegend=False
        )
        if output_chart:
            fig.update_xaxes(tickcolor='white', tickfont=dict(color='white'))
            fig.update_yaxes(tickcolor='white', tickfont=dict(color='white'))
            fig.update_layout(font=dict(color="white"))
            fig.write_image(str(output_dir / 'charts' / (U + '_' + V + '.png')), scale=resolution // 72)
        
        
    else:
        sns.violinplot(df[D], df[C])
        if len(df[D]) < 500:
            sns.swarmplot(x=df[D], y=df[C], edgecolor="white",
                          linewidth=1)  # Only show a swarm plot if there are fewer than 500 data points
        plt.xlabel(D.replace('_', ' ').title())
        plt.ylabel(C.replace('_', ' ').title())

        if output_chart:
            plt.savefig(output_dir / 'charts' / (U + '_' + V + '.png'), dpi=resolution)

    plt.close('all')

# Continuous-Continuous KDE Plots
def CC_viz(df, charter='Plotly', output_chart=False, output_dir=None, resolution=150):
    ''' Takes two continuous feature names and generates a 2D Kernel Density Plot '''
    U = list(df.columns)[0]
    V = list(df.columns)[1]

    if charter == 'Plotly':
        fig = ff.create_2d_density(df[U], df[V], colorscale=px.colors.sequential.Blues_r,
                                   hist_color=(135 / 255, 206 / 255, 250 / 255), title='')
        fig.update_layout(
            xaxis_showgrid=False, xaxis_zeroline=False,
            xaxis_title=U.replace('_', ' ').title(),
            yaxis_title=V.replace('_', ' ').title(),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            showlegend=False
        )

        if output_chart:
            fig.update_xaxes(tickcolor='white', tickfont=dict(color='white'))
            fig.update_yaxes(tickcolor='white', tickfont=dict(color='white'))
            fig.update_layout(font=dict(color="white"))
            fig.write_image(str(output_dir / 'charts' / (U + '_' + V + '.png')), scale=resolution // 72)
    else:
        sns.kdeplot(df[U], df[V], color='blue', shade=True, alpha=0.3, shade_lowest=False)
        if len(df[U]) < 500:
            sns.scatterplot(x=df[U], y=df[V], color='blue', alpha=0.5,
                            linewidth=0)  # Only show a scatter plot if there are fewer than 500 data points

        plt.xlabel(U.replace('_', ' ').title())
        plt.ylabel(V.replace('_', ' ').title())
        if output_chart:
            plt.savefig(output_dir / 'charts' / (U + '_' + V + '.png'), dpi=resolution)
            

    plt.close('all')


# Matrix Heatmap
def matrix_viz(matrix, output_chart=False, output_dir=None, resolution=150):
    plt.clf()
    plt.figure(dpi=70, figsize=(10, 8))
    sns.heatmap(matrix.fillna(0))
    if output_chart:
        plt.savefig(output_dir / 'heatmap.png', dpi=resolution)


# Visualization Function Router
def viz(U, V, df, feature_info, charter='Plotly', output_chart=False, output_json=False, output_dir=None, resolution=150):
    ''' Generate a visualization based on feature types '''
    plt.clf()
    plt.figure(dpi=resolution)
    pairdf = df.filter([U, V]).dropna(how='any')
    feature_types = feature_info['type'].to_dict()
    # If both features are discrete:
    if feature_types[U] == 'd' and feature_types[V] == 'd':
        DD_viz(pairdf, charter=charter, output_chart=output_chart, output_dir=output_dir, resolution=resolution)
    # If both features are continuous:
    elif feature_types[U] == 'c' and feature_types[V] == 'c':
        CC_viz(pairdf, charter=charter, output_chart=output_chart, output_dir=output_dir, resolution=resolution)
    # If one feature is continuous and one feature is discrete:
    elif feature_types[U] == 'c' and feature_types[V] == 'd':
        DC_viz(pairdf, charter=charter, output_chart=output_chart, output_dir=output_dir, resolution=resolution, discrete_first=False)
    elif feature_types[U] == 'd' and feature_types[V] == 'c':
        DC_viz(pairdf, charter=charter, output_chart=output_chart, output_dir=output_dir, resolution=resolution)
    else:
        raise Exception('Error on features', U, 'and', V)
        
    if output_json:
        with open(output_dir / 'json' / (U + '_' + V + '.json'), 'w') as outfile:
                json.dump(pairdf.to_dict(), outfile)

    return viz


def calculate_positions(G):
    # Generate position data for each node
    pos = nx.kamada_kawai_layout(G, weight='weight')

    # Save x, y locations of each edge
    edge_x = []
    edge_y = []

    # Calculate x,y positions of an edge's 'start' (x0,y0) and 'end' (x1,y1) points
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    # Bundle it all up in a dict:
    edges = dict(x=edge_x, y=edge_y)

    # Save x, y locations of each node
    node_x = []
    node_y = []

    # Save node stats for annotation
    node_name = []
    node_adjacencies = []
    node_centralities = []

    # Calculate x,y positions of nodes
    for node in G.nodes():
        node_name.append(node)  # Save node names
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))

    for n in G.nodes():
        node_centralities.append(nx.degree_centrality(G)[n])

    # Bundle it all up in a dict:
    nodes = dict(x=node_x, y=node_y, name=node_name, adjacencies=node_adjacencies, centralities=node_centralities)

    return edges, nodes


def draw_graph(stack, title, output_chart=False, output_dir=None, resolution=150, **kwargs):
    G = nx.from_pandas_edgelist(stack, 'x', 'y', ['v'])
    [edges, nodes] = calculate_positions(G)

    # Draw edges
    edge_trace = go.Scatter(
        x=edges['x'], y=edges['y'],
        line=dict(width=0.5, color='#888'),
        mode='lines+markers',
        hoverinfo='text')

    # Draw nodes
    node_trace = go.Scatter(
        x=nodes['x'],
        y=nodes['y'],
        mode='markers+text',
        text=nodes['name'],
        hoverinfo='text')
    filename = title.lower().replace(" ", "_")

    # Color the node by its number of connections
    # node_trace.marker.color = nodes['adjacencies']
    node_trace.marker.color = nodes['centralities']

    # Draw figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=120),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template='plotly_white')
    )

    fig.update_traces(textposition='top center')

    if output_chart:
        fig.write_image(str(output_dir / 'graph_example.png'), scale=resolution // 72)
