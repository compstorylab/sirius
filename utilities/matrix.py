

import itertools
from datetime import datetime
import pandas as pd
import numpy as np
import math
import random
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.metrics import mutual_info_score
from scipy.stats import multivariate_normal, pearsonr
import scipy.integrate as integrate
from sklearn.neighbors import KernelDensity
from pathlib import Path
import networkx as nx
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.colors import n_colors
# If you're using this code locally:
from plotly.offline import download_plotlyjs, plot, iplot #, init_notebook_mode


def load_data(input_file, sample_n=None, debug=False):
    # Load Data
    df = pd.read_csv(input_file)
    if sample_n:
        df = df.sample(sample_n)

    # Ignore ID columns
    ignore = []
    for i in df.columns:
        try:
            # ? What is this doing?
            np.ma.fix_invalid(df[i])
        except:
            pass

        if ('_id' in i):
            ignore.append(i)

    df = df.drop(columns=ignore)
    df = df.replace(np.nan, None)
    df = df.replace('nan', None)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    if debug:
        print(f'Loaded data from {input_file} with {df.shape[0]} observations and {df.shape[1]} features')

    return df


## Visualization

def compute_bandwidth(X, df):
    ''' Takes a column name and computes suggested gaussian bandwidth with the formula: 1.06*var(n^-0.2) '''
    var = np.var(df[X])
    n = len(df[X].notnull())
    b = 1.06 * var * (n ** (-0.2))
    return b


# Discrete-Discrete Confusion Matrices
def DD_viz(df, charter='Plotly', chart=False, output=False, output_dir=None, resolution=150):
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
        if chart:
            fig.show()

        if output:
            fig.update_xaxes(tickcolor='white', tickfont=dict(color='white'))
            fig.update_yaxes(tickcolor='white', tickfont=dict(color='white'))
            fig.update_layout(font=dict(color="white"))
            with open(output_dir / 'charts' / (U + '_' + V + '.json'), 'w') as outfile:
                json.dump(fig.to_json(), outfile)

            fig.write_image(str(output_dir / 'charts' / (U + '_' + V + '.png')), scale=resolution // 72)
    else:
        plt.clf()
        plt.figure(dpi=resolution)
        sns.heatmap(s, annot=True, cmap="Blues", cbar=False, linewidths=1)
        plt.xlabel(U.replace('_', ' ').title())
        plt.ylabel(V.replace('_', ' ').title())
        if output:
            plt.savefig(output_dir / 'charts' / (U + '_' + V + '.png'), dpi=resolution)

        if chart:
            plt.show()

    plt.close('all')


# Discrete-Continuous Violin Plots
def DC_viz(df, charter='Plotly', chart=False, output=False, output_dir=None, resolution=150, discrete_first=True):
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
        if chart:
            fig.show()

        if output:
            fig.update_xaxes(tickcolor='white', tickfont=dict(color='white'))
            fig.update_yaxes(tickcolor='white', tickfont=dict(color='white'))
            fig.update_layout(font=dict(color="white"))
            with open(output_dir / 'charts' / (U + '_' + V + '.json'), 'w') as outfile:
                json.dump(fig.to_json(), outfile)

            fig.write_image(str(output_dir / 'charts' / (U + '_' + V + '.png')), scale=resolution // 72)
    else:
        sns.violinplot(df[D], df[C])
        if len(df[D]) < 500:
            sns.swarmplot(x=df[D], y=df[C], edgecolor="white",
                          linewidth=1)  # Only show a swarm plot if there are fewer than 500 data points
        plt.xlabel(D.replace('_', ' ').title())
        plt.ylabel(C.replace('_', ' ').title())

        if output:
            plt.savefig(output_dir / 'charts' / (U + '_' + V + '.png'), dpi=resolution)

        if chart:
            plt.show()

    plt.close('all')


# Continuous-Continuous KDE Plots
def CC_viz(df, charter='Plotly', chart=False, output=False, output_dir=None, resolution=150):
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
        if chart:
            fig.show()

        if output:
            fig.update_xaxes(tickcolor='white', tickfont=dict(color='white'))
            fig.update_yaxes(tickcolor='white', tickfont=dict(color='white'))
            fig.update_layout(font=dict(color="white"))
            with open(output_dir / 'charts' / (U + '_' + V + '.json'), 'w') as outfile:
                json.dump(fig.to_json(), outfile)

            fig.write_image(str(output_dir / 'charts' / (U + '_' + V + '.png')), scale=resolution // 72)
    else:
        sns.kdeplot(df[U], df[V], color='blue', shade=True, alpha=0.3, shade_lowest=False)
        if len(df[U]) < 500:
            sns.scatterplot(x=df[U], y=df[V], color='blue', alpha=0.5,
                            linewidth=0)  # Only show a scatter plot if there are fewer than 500 data points

        plt.xlabel(U.replace('_', ' ').title())
        plt.ylabel(V.replace('_', ' ').title())
        if output:
            plt.savefig(output_dir / 'charts' / (U + '_' + V + '.png'), dpi=resolution)

        if chart:
            plt.show()

    plt.close('all')


# Matrix Heatmap
def matrix_viz(matrix):
    plt.clf()
    plt.figure(dpi=70, figsize=(10, 8))
    sns.heatmap(matrix.fillna(0))
    plt.show()


def output_pairs_json(df, output_dir, pairs=None):
    if pairs is None:
        pairs = list(itertools.combinations(df.columns, 2))

    for col1, col2 in pairs:
        pairdf = df.loc[:, [col1, col2]].dropna(how='any')
        pairdf.to_json(output_dir / 'json' / (col1 + '_' + col2 + '.json'))


# Visualization Function Router
def viz(U, V, df, feature_types, charter='Plotly', chart=False, output=False, output_dir=None, resolution=150):
    ''' Generate a visualization based on feature types '''
    plt.clf()
    plt.figure(dpi=resolution)
    pairdf = df.filter([U, V]).dropna(how='any')

    # If both features are discrete:
    if feature_types[U] == 'd' and feature_types[V] == 'd':
        DD_viz(pairdf, charter=charter, chart=chart, output=output, output_dir=output_dir, resolution=resolution)
    # If both features are continuous:
    elif feature_types[U] == 'c' and feature_types[V] == 'c':
        CC_viz(pairdf, charter=charter, chart=chart, output=output, output_dir=output_dir, resolution=resolution)
    # If one feature is continuous and one feature is discrete:
    elif feature_types[U] == 'c' and feature_types[V] == 'd':
        DC_viz(pairdf, charter=charter, chart=chart, output=output, output_dir=output_dir, resolution=resolution, discrete_first=False)
    elif feature_types[U] == 'd' and feature_types[V] == 'c':
        DC_viz(pairdf, charter=charter, chart=chart, output=output, output_dir=output_dir, resolution=resolution)
    else:
        raise Exception('Error on features', U, 'and', V)

    return viz


## Feature Classification

def classify_features(df, discrete_threshold, debug=False):
    """
    Categorize every feature/column of df as discrete or continuous according to whether or not the unique responses
    are numeric and, if they are numeric, whether or not there are fewer unique renponses than discrete_threshold.
    Return a dataframe with a row for each feature and columns for the type, the count of unique responses, and the
    count of string, number or null/nan responses.
    """
    counts = []
    string_counts = []
    float_counts = []
    null_counts = []
    types = []
    for col in df.columns:
        responses = df[col].unique()
        counts.append(len(responses))
        string_count, float_count, null_count = 0, 0, 0
        for value in responses:
            try:
                val = float(value)
                if not math.isnan(val):
                    float_count += 1
                else:
                    null_count += 1
            except ValueError:
                try:
                    val = str(value)
                    string_count += 1
                except:
                    print('Error: Unexpected value', value, 'for feature', col)

        string_counts.append(string_count)
        float_counts.append(float_count)
        null_counts.append(null_count)
        types.append('d' if len(responses) < discrete_threshold or string_count > 0 else 'c')

    feature_info = pd.DataFrame({'count': counts,
                                 'string_count': string_counts,
                                 'float_count': float_counts,
                                 'null_count': null_counts,
                                 'type': types}, index=df.columns)
    if debug:
        print(f'Counted {sum(feature_info["type"] == "d")} discrete features and {sum(feature_info["type"] == "c")} continuous features')

    return feature_info


## Mutual Information

def calc_mi(df, feature_info, debug=False):
    """
    Calculate the mutual information between every pair of features in df.
    Return a dataframe whose columns are 'x' (the first column in the pair), 'y' (the second column),
    and 'v' (the mutual information of the pair). Each row is a distinct pair of variables and the MI between
    them. Features with only one response (unary features) are ignored.
    df: a data frame whose rows are observations and columns are features.
    feature_info: a dataframe with a row for each feature (in the same order as the columns of dm) and a 'type'
    column that classifies each feature as 'c' (continuous) or 'd' (discrete).
    """
    start_time = datetime.now()

    # drop unary features
    unary_cols = feature_info.index[feature_info['count'] <= 1]
    df = df.drop(columns=unary_cols)
    feature_info = feature_info.drop(index=unary_cols)

    # Convert df into a numeric matrix for mutual information calculations
    formatted_df = df.copy()
    for col in formatted_df.columns:
        if feature_info.loc[col, 'type'] == 'c':
            formatted_df[col] = formatted_df[col].map(float)
        elif feature_info.loc[col, 'type'] == 'd':
            # mutual_info_regression expects numeric values for discrete columns
            formatted_df[col] = formatted_df[col].factorize()[0]
        else:
            raise Exception('Error formatting column ', col)

    dm = formatted_df.values  # data matrix (rows: observations, columns: features)

    num_features = dm.shape[1]
    pairs = list(itertools.combinations(range(num_features), 2))  # all pairs of features (as indices)
    xs, ys, vs = [], [], []
    for i, j in pairs:
        pair_start_time = datetime.now()
        x_name = feature_info.index[i]
        y_name = feature_info.index[j]
        if debug:
            print(f'Calculating MI for {x_name} ({i} of {num_features}) and {y_name} ({j} of {num_features})')

        nan_idx = np.isnan(dm[:, i]) | np.isnan(dm[:, j])  # boolean index of the rows which contain a nan value
        u = dm[~nan_idx, i]  # select observations for which both features are present
        v = dm[~nan_idx, j]
        if len(u) == 0:
            mi = 0  # no shared observations -> no mutual information
        elif feature_info['type'][i] == 'c':  # continuous
            if feature_info['type'][j] == 'c':
                # mutual_info_regression expects a 2d matrix of features
                mi = mutual_info_regression(u.reshape(-1, 1), v, discrete_features=False)[0]
            else:  # v is discrete
                mi = mutual_info_regression(v.reshape(-1, 1), u, discrete_features=True)[0]
        else:  # u is discrete
            if feature_info['type'][j] == 'c':
                mi = mutual_info_regression(u.reshape(-1, 1), v, discrete_features=True)[0]
            else:  # v is discrete
                mi = np.log2(np.e) * mutual_info_score(u, v)  # convert from base e to base 2

        if debug:
            print(f'Elapsed time:', datetime.now() - pair_start_time)

        xs.append(x_name)
        ys.append(y_name)
        vs.append(mi)

    if debug:
        print('Elapsed time:', datetime.now() - start_time)
        print('Calculated mutual information for', num_features, 'columns across', dm.shape[0], 'records')

    return pd.DataFrame({'x': xs, 'y': ys, 'v': vs})


## Network Graph

def output_graph_json(stack, feature_types, output_dir):
    # Create a networkx graph from the list of pairs
    G = nx.from_pandas_edgelist(stack, 'source', 'target', ['weight'])
    nodelist = []
    for n in set(stack['source']).union(stack['target']):
        nodelist.append({'name': n,
                         'type': 'continuous' if feature_types[n] == 'c' else 'discrete',
                         'neighbors': list(dict(G[n]).keys())})

    json_out = {'nodes': nodelist, 'links': stack.to_dict(orient='records')}
    with open(output_dir / 'graph.json', 'w') as json_file:
        json.dump(json_out, json_file)

    components = []
    for i in nx.connected_components(G):
        components.append(list(i))

    with open(output_dir / 'components.json', 'w') as json_file:
        json.dump(components, json_file)


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


def draw_graph(stack, title, chart=False, output=False, output_dir=None, resolution=150, **kwargs):
    G = nx.from_pandas_edgelist(stack, 'source', 'target', ['weight'])
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
        # Optional: Add labels to points *without* hovering (can get a little messy)
        mode='markers+text',
        # ...or, just add markers (no text)
        # mode='markers',
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
    # Show figure
    if chart:
        fig.show()

    if output:
        fig.write_image(str(output_dir / 'graph_example.png'), scale=resolution // 72)


## Thresholding

def find_max_component_threshold(stack):
    """
    Find the mutual information threshold that maximizes the number of connected components (subgraphs with >1 node)
    in the resulting graph.
    :param stack: every row is a pair of features and their associated mutual information.
    :return: the mutual information threshold.
    """

    # Create a data frame of edge counts and number of components for a given threshold
    e = pd.DataFrame(columns=['mi_threshold', 'edge_count', 'components'])

    # Fill in the 'e' data frame with the number of edges and number of components across a range of thresholds
    for i in np.arange(np.round(stack['v'].min(), 2), np.round(stack['v'].max(), 2), 0.01):
        s = stack[stack['v'] > i]

        G = nx.Graph()
        G.add_nodes_from(list(dict.fromkeys((list(s['x'].unique()) + list(s['y'].unique())))))
        G.add_edges_from(list(zip(s['x'], s['y'])))

        e = e.append({'mi_threshold': i, 'edge_count': (stack['v'] > i).sum(),
                      'components': nx.number_connected_components(G)}, ignore_index=True)

    # Plot the number of edges for a range of mutual information scores
    sns.lineplot(e['mi_threshold'], e['edge_count'])

    # Plot the number of components for a range of mutual information scores
    sns.lineplot(e['mi_threshold'], e['components'])

    # Find the mutual information threshold which maximizes the component count
    max_component_threshold = e[e['components'] == max(e['components'])].max()['mi_threshold']
    # optional: if there are multiple MI thresholds which maximize the number of components,
    # you may want to experiment with thresholding at .min()['mi_threshold'] instead of .max()['mi_threshold']
    # depending on your application; we have selected the maximum threshold for maximizing component counts,
    # as this minimizes the edge count over the minimum threshold, which reduces the visualization exploration space,
    # while still maximizing component counts

    return max_component_threshold


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Sirius Data Processing Pipeline')
    parser.add_argument('--dpi', type=int, default=150, help='resolution of output plots')
    parser.add_argument('--discrete-threshold', type=int, default=5,
                        help='Number of responses below which numeric features are considered discrete')
    parser.add_argument('--chart', action='store_true', default=False, help='Display images while running computation')
    parser.add_argument('--charter', choices=['Plotly', 'Seaborn'], default='Plotly', help='The plotting library to use.')
    parser.add_argument('--debug', action='store_true', default=False, help='Print updates to the console while running.')
    parser.add_argument('--output', action='store_true', default=False, help='Output network and feature pairs json to files.')
    parser.add_argument('--output-chart', action='store_true', default=False, help='Output chart json and pngs to files.')
    parser.add_argument('--no-viz', action='store_true', default=False, help='Do not output pair plots, network graph, or json.')
    parser.add_argument('--no-mi', action='store_true', default=False, help='Do not compute MI. Use cached MI values instead.')
    parser.add_argument('--cache', action='store_true', default=False, help='Cache MI values to use later when generating visualizations.')
    parser.add_argument('--sample-n', default=None, type=int, help='Subsample the data. By default, work with all the data.')
    parser.add_argument('--input-file', default='example_data/data.csv', help='Location of the input CSV data.')
    parser.add_argument('--output-dir', default='example_data/output', help='A directory in which to store the json and png files.')
    args = parser.parse_args()

    # Parameter settings
    chart = args.chart  # boolean for whether to display images while running computation
    debug = args.debug  # boolean for whether to print updates to the console while running
    output = args.output  # boolean for whether to output json and pngs to files
    output_chart = args.output_chart  # boolean for whether to output json and pngs to files
    cache = args.cache
    no_mi = args.no_mi
    no_viz = args.no_viz
    charter = args.charter  # accepts 'Seaborn' or 'Plotly'
    resolution = args.dpi  # int for resolution of output plots
    discrete_threshold = args.discrete_threshold  # number of responses below which numeric responses are considered discrete
    compare_all = True  # boolean; if comparing two lists of the same length, fill in list1 and list2 accordingly
    list1, list2 = [], []
    sample_n = args.sample_n  # Work with all data (None), or just a sample?
    input_file = Path(args.input_file)  # e.g. './example_data/data.csv'
    output_dir = Path(args.output_dir)  # e.g. './example_data/output'
    # cd = 'example_data/output'

    sns.set_style("whitegrid")

    # create output directories, if needed
    if cache:
        output_dir.mkdir(parents=True, exist_ok=True)

    if output:
        (output_dir / 'json').mkdir(parents=True, exist_ok=True)

    if output_chart:
        (output_dir / 'charts').mkdir(parents=True, exist_ok=True)

    # Load Data
    df = load_data(input_file, sample_n=sample_n, debug=debug)

    # Classify Features
    feature_info = classify_features(df, discrete_threshold, debug=debug)
    feature_types = feature_info['type'].to_dict()
    if debug:
        print('Classified Features:')
        print(feature_info)

    if not no_mi:
        stack = calc_mi(df, feature_info, debug=debug)
        if cache:
            stack.to_csv(output_dir / 'results.csv', index=False)

    # Network Graphing

    if no_viz:
        return

    if cache:
        # Re-import the Mutual Information results
        # (this is helpful if you want to re-generate visualizations
        # without having to re-run the mutual information calculations)
        stack = pd.read_csv(output_dir / 'results.csv')

    # Sort our values and (optionally) exclude Mutual Infomation scores above 1 (which are often proxies for one another)
    sorted_stack = stack.sort_values(by='v', ascending=False)
    # sorted_stack = sorted_stack[sorted_stack['v'] < 1]

    # Thresholding

    max_component_threshold = find_max_component_threshold(sorted_stack)
    # Threshold the edge list by the mutual information threshold which maximizes the component count
    thresh_stack = sorted_stack[sorted_stack['v'] > max_component_threshold]

    # Add visualization type to each pair: CC, DC, or DD
    thresh_stack['viztype'] = thresh_stack.apply(
        lambda s: ''.join(sorted([feature_info.loc[s['x'], 'type'], feature_info.loc[s['y'], 'type']],
                                 reverse=True)).upper(),
        axis=1)

    thresh_stack = thresh_stack.rename(columns={'x': 'source', 'y': 'target', 'v': 'weight'})

    # Network Graph

    if output:
        output_graph_json(thresh_stack, feature_types, output_dir)

    if output_chart or chart:
        draw_graph(thresh_stack, 'Example Graph', chart=chart, output=output_chart, output_dir=output_dir, resolution=resolution)

    # Feature Pairs

    if output:
        output_pairs_json(df, output_dir, pairs=list(zip(thresh_stack['source'], thresh_stack['target'])))

    if output_chart or chart:
        for i, row in thresh_stack.iterrows():
            viz(row['source'],row['target'], df, feature_types, charter=charter, chart=chart, output=output_chart,
                output_dir=output_dir, resolution=resolution)


if __name__ == '__main__':
    import cProfile
    cProfile.run('main()')
    # main()



