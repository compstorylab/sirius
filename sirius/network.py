## Network Graph

import pandas as pd
import numpy as np
import networkx as nx
from scipy import integrate
import itertools
import json
import seaborn as sns
sns.set_style("whitegrid")


def output_pairs_json(df, output_dir, pair_info=None, num_sample=None, random_state=None):
    """
    :param df:
    :param output_dir:
    :param pair_info: 'x' and 'y' columns specify feature pairs. 'v' column specifies mutual information.
    :param num_sample:
    :param random_state:
    :return:
    """
    json_dir = output_dir / 'json'
    json_dir.mkdir(parents=True, exist_ok=True)
    if pair_info is not None:
        pairs = list(zip(pair_info['x'], pair_info['y']))
    else:
        pairs = list(itertools.combinations(df.columns, 2))

    for col1, col2 in pairs:
        pairdf = df.loc[:, [col1, col2]].dropna(how='any')
        if num_sample and len(pairdf) > num_sample:
            # too many data points make the javascript visualizations cry.
            pairdf = pairdf.sample(n=num_sample, random_state=random_state)

        pairdf.to_json(json_dir / (col1 + '_' + col2 + '.json'))


def output_graph_json(pair_info, feature_info, output_dir):
    """
    :param pair_info: 'x' and 'y' columns specify feature pairs. 'v' column specifies mutual information.
    :param feature_info: indexed by feature. 'type' column specifies 'c' for continuous and 'd' for discrete
    :param output_dir:
    :return:
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Add visualization type to each pair: CC, DC, or DD
    pair_info['viztype'] = pair_info.apply(
        lambda s: ''.join(sorted([feature_info.loc[s['x'], 'type'], feature_info.loc[s['y'], 'type']],
                                 reverse=True)).upper(),
        axis=1)

    # Rename columns for Sirius
    pair_info = pair_info.rename(columns={'x': 'source', 'y': 'target', 'v': 'weight'})

    # Create a networkx graph from the list of pairs
    G = nx.from_pandas_edgelist(pair_info, 'source', 'target', ['weight'])
    nodelist = []
    for n in set(pair_info['source']).union(pair_info['target']):
        nodelist.append({'name': n,
                         'type': 'continuous' if feature_info.loc[n, 'type'] == 'c' else 'discrete',
                         'neighbors': list(dict(G[n]).keys())})

    json_out = {'nodes': nodelist, 'links': pair_info.to_dict(orient='records')}
    with open(output_dir / 'graph.json', 'w') as json_file:
        json.dump(json_out, json_file)

    components = []
    for i in nx.connected_components(G):
        components.append(list(i))

    with open(output_dir / 'components.json', 'w') as json_file:
        json.dump(components, json_file)
        
## Dynamic Thresholding Using Backbone Method

def disparity_filter(G, weight='weight'):
    ''' Compute significance scores (alpha) for weighted edges in G as defined in Serrano et al. 2009
        References:
            Palakorn Achananuparp, "Python Backbone Network": https://github.com/aekpalakorn/python-backbone-network
            M. A. Serrano et al. (2009) Extracting the Multiscale backbone of complex weighted networks. PNAS, 106:16, pp. 6483-6488.
    '''
    B = nx.Graph()
    for u in G:
        k = len(G[u])
        if k > 1:
            sum_w = sum(np.absolute(G[u][v][weight]) for v in G[u])
            for v in G[u]:
                w = G[u][v][weight]
                p_ij = float(np.absolute(w))/sum_w
                alpha_ij = 1 - (k-1) * integrate.quad(lambda x: (1-x)**(k-2), 0, p_ij)[0]
                B.add_edge(u, v, weight = w, alpha=float('%.4f' % alpha_ij))
    return B

def threshold_using_backbone_method(pair_info):
    G = nx.Graph()
    G.add_nodes_from(list(dict.fromkeys((list(pair_info['x'].unique()) + list(pair_info['y'].unique())))))
    G.add_weighted_edges_from(list(zip(pair_info['x'], pair_info['y'],pair_info['v'])))
    alpha = 0.05
    G = disparity_filter(G)
    G2 = nx.Graph([(u, v, d) for u, v, d in G.edges(data=True) if d['alpha'] < alpha])
    thresheld_edgelist = nx.to_pandas_edgelist(G2).drop(columns='alpha').rename(columns={'source':'x','target':'y','weight':'v'})
    return thresheld_edgelist


## Static Thresholding (deprecated)

def threshold_by_max_component(pair_info, output_chart=False, resolution=150):
    max_component_threshold = find_max_component_threshold(pair_info, output_chart=output_chart, resolution=resolution)
    # Threshold the edge list by the mutual information threshold which maximizes the component count
    return pair_info[pair_info['v'] > max_component_threshold]


def find_max_component_threshold(stack, output_chart=False, resolution=150):
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

    if output_chart:
        # Plot the number of edges for a range of mutual information scores
        sns.lineplot(e['mi_threshold'], e['edge_count'])
        #plt.savefig(output_dir / 'mi_vs_edge-count.png', dpi=args['resolution'])
        #plt.close('all')
        # Plot the number of components for a range of mutual information scores
        sns.lineplot(e['mi_threshold'], e['components'])
        #plt.savefig(output_dir / 'mi_vs_components.png', dpi=args['resolution'])
        #plt.close('all')
        
    # Find the mutual information threshold which maximizes the component count
    max_component_threshold = e[e['components'] == max(e['components'])].max()['mi_threshold']
    # optional: if there are multiple MI thresholds which maximize the number of components,
    # you may want to experiment with thresholding at .min()['mi_threshold'] instead of .max()['mi_threshold']
    # depending on your application; we have selected the maximum threshold for maximizing component counts,
    # as this minimizes the edge count over the minimum threshold, which reduces the visualization exploration space,
    # while still maximizing component counts

    return max_component_threshold
