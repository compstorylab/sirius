import itertools
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import seaborn as sns
# If you're using this code locally:
# from plotly.offline import download_plotlyjs, iplot, plot  # , init_notebook_mode
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)


def get_types(U, response_list):
    types = {"floats": 0, "strings": 0, "nulls": 0}
    for i in response_list["responses"][U]:
        try:
            val = float(i)
            if not math.isnan(val):
                # print("Value",i," is a float")
                types["floats"] += 1
            else:
                # print("Value",i," is null")
                types["nulls"] += 1
        except ValueError:
            try:
                val = str(i)
                # print("Value",i,"is a string")
                types["strings"] += 1
            except:
                logging.warning("Error: Unexpected value", i, "for feature", U)

    if types["floats"] > 0 and types["strings"] > 0:
        logging.warning("Column", U, "contains floats AND strings")

    return types


def compute_bandwidth(X, df):
    """
    Takes a column name and computes suggested gaussian bandwidth with the formula:
        1.06 * var(n^-0.2)

    TODO: Unused.
    """
    var = np.var(df[X])
    n = len(df[X].notnull())
    b = 1.06 * var * (n ** (-0.2))
    return b


def DD_viz(
        df,
        charter="Plotly",
        chart=False,
        output=False,
        output_dir=None,
        resolution=150,
):
    """
    Takes a filtered dataframe of two discrete feature columns and generates a heatmap
    """

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

    if charter == "Plotly":
        fig = ff.create_annotated_heatmap(
            s.values,
            x=[str(i) for i in i_range],
            y=[str(j) for j in j_range],
            colorscale="Blues",
        )
        fig.update_layout(
            xaxis_title=U.replace("_", " ").title(),
            yaxis_title=V.replace("_", " ").title(),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
        )
        if chart:
            fig.show()

        if output:
            fig.update_xaxes(tickcolor="white", tickfont=dict(color="white"))
            fig.update_yaxes(tickcolor="white", tickfont=dict(color="white"))
            fig.update_layout(font=dict(color="white"))
            with open(output_dir / "charts" / U + "_" + V + ".json", "w") as outfile:
                json.dump(fig.to_json(), outfile)

            fig.write_image(
                output_dir / "charts" / U + "_" + V + ".png", scale=resolution // 72
            )
    else:
        plt.clf()
        plt.figure(dpi=resolution)
        sns.heatmap(s, annot=True, cmap="Blues", cbar=False, linewidths=1)
        plt.xlabel(U.replace("_", " ").title())
        plt.ylabel(V.replace("_", " ").title())
        if output:
            plt.savefig(output_dir / "charts" / U + "_" + V + ".png", dpi=resolution)

        if chart:
            plt.show()

        plt.close("all")


def DC_viz(
        df,
        continuous,
        charter="Plotly",
        chart=False,
        output=False,
        output_dir=None,
        resolution=150,
):
    """ Takes a subset dataframe of one continuous and one discrete feature and generates a Violin Plot """

    U = list(df.columns)[0]
    V = list(df.columns)[1]

    if U in continuous:
        D = V
        C = U
    else:
        D = U
        C = V

    if charter == "Plotly":
        fig = go.Figure()
        for i in list(df[D].unique()):
            series = df[df[D] == i][C]
            fig.add_trace(go.Violin(x=series, name=str(i)))

        fig.update_traces(orientation="h", side="positive", width=3, points=False)
        fig.update_layout(
            xaxis_showgrid=False,
            xaxis_zeroline=False,
            xaxis_title=U.replace("_", " ").title(),
            yaxis_title=V.replace("_", " ").title(),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            showlegend=False,
        )
        if chart:
            fig.show()

        if output:
            fig.update_xaxes(tickcolor="white", tickfont=dict(color="white"))
            fig.update_yaxes(tickcolor="white", tickfont=dict(color="white"))
            fig.update_layout(font=dict(color="white"))
            with open(output_dir / "charts" / U + "_" + V + ".json", "w") as outfile:
                json.dump(fig.to_json(), outfile)

            fig.write_image(
                output_dir / "charts" / U + "_" + V + ".png", scale=resolution // 72
            )
    else:
        sns.violinplot(df[D], df[C])
        if len(df[D]) < 500:
            sns.swarmplot(
                x=df[D], y=df[C], edgecolor="white", linewidth=1
            )  # Only show a swarm plot if there are fewer than 500 data points
        plt.xlabel(D.replace("_", " ").title())
        plt.ylabel(C.replace("_", " ").title())

        if output:
            plt.savefig(output_dir / "charts" / U + "_" + V + ".png", dpi=resolution)

        if chart:
            plt.show()

    plt.close("all")


def CC_viz(
        df, charter="Plotly", chart=False, output=False, output_dir=None, resolution=150,
):
    """ Takes two continuous feature names and generates a 2D Kernel Density Plot """
    U = list(df.columns)[0]
    V = list(df.columns)[1]

    if charter == "Plotly":
        fig = ff.create_2d_density(
            df[U],
            df[V],
            colorscale=px.colors.sequential.Blues_r,
            hist_color=(135 / 255, 206 / 255, 250 / 255),
            title="",
        )
        fig.update_layout(
            xaxis_showgrid=False,
            xaxis_zeroline=False,
            xaxis_title=U.replace("_", " ").title(),
            yaxis_title=V.replace("_", " ").title(),
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            showlegend=False,
        )
        if chart:
            fig.show()

        if output:
            fig.update_xaxes(tickcolor="white", tickfont=dict(color="white"))
            fig.update_yaxes(tickcolor="white", tickfont=dict(color="white"))
            fig.update_layout(font=dict(color="white"))
            with open(output_dir / "charts" / U + "_" + V + ".json", "w") as outfile:
                json.dump(fig.to_json(), outfile)

            fig.write_image(
                output_dir / "charts" / U + "_" + V + ".png", scale=resolution // 72
            )
    else:
        sns.kdeplot(
            df[U], df[V], color="blue", shade=True, alpha=0.3, shade_lowest=False
        )
        if len(df[U]) < 500:
            sns.scatterplot(
                x=df[U], y=df[V], color="blue", alpha=0.5, linewidth=0
            )  # Only show a scatter plot if there are fewer than 500 data points

        plt.xlabel(U.replace("_", " ").title())
        plt.ylabel(V.replace("_", " ").title())
        if output:
            plt.savefig(output_dir / "charts" / U + "_" + V + ".png", dpi=resolution)

        if chart:
            plt.show()

    plt.close("all")


def matrix_heatmap(matrix):
    # TODO: Unused.
    plt.clf()
    plt.figure(dpi=70, figsize=(10, 8))
    sns.heatmap(matrix.fillna(0))
    plt.show()


# Visualization Function Router
def viz(
        U,
        V,
        df,
        discrete,
        continuous,
        charter="Plotly",
        chart=False,
        output=False,
        output_dir=None,
        resolution=150,
):
    """ Generate a visualization based on feature types """
    plt.clf()
    plt.figure(dpi=resolution)

    pairdf = df.filter([U, V]).dropna(how="any")

    # If both features are discrete:
    if U in discrete and V in discrete:
        DD_viz(
            pairdf,
            charter=charter,
            chart=chart,
            output=output,
            output_dir=output_dir,
            resolution=resolution,
        )
    # If both features are continuous:
    elif U in continuous and V in continuous:
        CC_viz(
            pairdf,
            charter=charter,
            chart=chart,
            output=output,
            output_dir=output_dir,
            resolution=resolution,
        )
    # If one feature is continuous and one feature is discrete:
    elif U in continuous and V in discrete or U in discrete and V in continuous:
        DC_viz(
            pairdf,
            continuous,
            charter=charter,
            chart=chart,
            output=output,
            output_dir=output_dir,
            resolution=resolution,
        )
    else:
        raise Exception("Error on features", U, "and", V)

    if output:
        pairdf.to_json(output_dir / "json" / U + "_" + V + ".json")

    return viz


# Mutual Information
def sparsify(series):
    """
    For discrete values: takes a column name and returns a sparse matrix (0 or 1)
    with a column for each unique response
    """
    responses = series.unique()
    m = pd.DataFrame(columns=responses)
    for val in responses:
        m[val] = series == val

    return m.astype(int)


def DD_mi(df, debug=False):
    """
    Takes two discrete feature names and calculates normalized mutual
    information (dividing mutual information by maximum possible)
    """
    U = list(df.columns)[0]
    V = list(df.columns)[1]

    logging.debug(f"Calculating discrete-discrete MI for {U} and {V}")

    min_response_count = min(len(list(df[U].unique())), len(list(df[V].unique())))
    max_mi = np.log2(min_response_count)
    if U == V:
        mi = max_mi
    else:
        i_range = list(df[U].unique())
        j_range = list(df[V].unique())
        # We use 's' to denote a matrix of support for each i,j
        s = pd.DataFrame(columns=i_range, index=j_range)
        for i in i_range:
            for j in j_range:
                s[i][j] = (
                    df[(df[U] == i) & (df[V] == j)].filter([U, V], axis=1).shape[0]
                )
                mutual_support = s.sum().sum()
        s = s.astype(int)
        pmi = s.copy()
        l = []
        # If these features are never both answered, or if either feature only has one possible response:
        if mutual_support <= 0 or len(i_range) <= 1 or len(j_range) <= 1:
            # The whole pointwise mutual information matrix should be 0
            pmi.fillna(0, inplace=True)
        else:
            for i in i_range:
                for j in j_range:
                    joint_support = s[i][j]
                    joint_probability = joint_support / mutual_support
                    marginal_probability_i = s.sum(axis=0)[i] / s.sum().sum()
                    marginal_probability_j = s.sum(axis=1)[j] / s.sum().sum()
                    if joint_probability != 0:
                        pmi[i][j] = np.log2(
                            joint_probability
                            / (marginal_probability_i * marginal_probability_j)
                        )
                        # Store all PMI (pointwise mutual information) in a list
                        l.append(pmi[i][j] * joint_probability)
            # Sum the list of all pointwise mutual information
            mi = sum(l)

    # Normalize by response count (not recommended)
    # if max_mi==0:
    #     nmi = 0
    # else:
    #     nmi = mi/max_mi

    logging.debug(f"MI: {mi}")

    return mi


def DC_mi(df, continuous, debug=False):
    """
    Calculates the mutual information between a discrete feature and a continuous feature.
    Uses a sparsified (one-hot) encoding of the discrete feature.

    Implemented with SciKit's [mutual_info_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html)
    which calculates mutual information using the nearest neighbor entropy approach described in
    [*B. C. Ross “Mutual Information between Discrete and Continuous Data Sets”. PLoS ONE 9(2), 2014.*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357)

    This requires us to sparsify the discrete matrix by response.
    """

    U = list(df.columns)[0]
    V = list(df.columns)[1]

    logging.debug(f"Calculating discrete-continious MI for {U} and {V}")

    if U in continuous:
        D = V
        C = U
    else:
        D = U
        C = V
    logging.debug(f"Discrete: {D} Continuous: {C}")

    responses = list(df[D].unique())
    logging.debug(f"responses = {responses}")

    pmi = list(mutual_info_regression(sparsify(df[D]), df[C], discrete_features=True))
    logging.debug(f"pmi list = {pmi}")

    l = []
    for i in list(range(0, (len(responses)))):
        conditional_probability = df[df[D] == responses[i]].shape[0] / len(df[C])
        l.append(pmi[i] * conditional_probability)

    mi = sum(l)
    logging.debug(f"MI: {mi}")

    return mi


# Continuous-Continuous
def CC_mi(df, debug=False):
    """
    Calculates the mutual information between two continuous features using
    SciKit's mutual_info_regression function.
    """
    U = list(df.columns)[0]
    V = list(df.columns)[1]

    logging.debug(f"Calculating mutual information for {U} and {V}")

    mi = mutual_info_regression(df.filter([U]), df[V])[0]
    logging.debug(f"MI: {mi}")

    return mi


def CC_corr(df, debug=False):
    """
    Calculates the Pearson's correlation between two continuous features.

    For use in qualitative comparisons with the mutual information implementation above.
    """

    U = list(df.columns)[0]
    V = list(df.columns)[1]

    logging.debug(f"Calculating correlation between {U} and {V}")

    corr = pearsonr(df[U], df[V])[0]
    logging.debug(f"Correlation = {corr}")

    return corr


def matrixify(df):
    """
    Takes a dataframe with columns [source,target,value] and returns a matrix where
    {index:source, columns:target, values:values}

    TODO: Unused.
    """
    m = df.pivot(
        index=list(df.columns)[0],
        columns=list(df.columns)[1],
        values=list(df.columns)[2],
    )
    return m


def calc_pairtype(U, V, discrete, continuous, debug=False):
    """
    Takes two feature names and returns the pair type
    ('DD': discrete/discrete, 'DC': discrete/continuous, or 'CC': continuous/continuous)
    """

    logging.debug('Finding pair type for "', U, '" and "', V, '"')

    # If both features are discrete:
    if U in discrete and V in discrete:
        pair_type = "DD"
        logging.debug('"', U, '" and "', V, '" are data pair type', pair_type)
    # If both features are continuous:
    elif U in continuous and V in continuous:
        pair_type = "CC"
        logging.debug('"', U, '" and "', V, '" are data pair type', pair_type)
    # If one feature is continuous and one feature is discrete:
    elif U in continuous and V in discrete or U in discrete and V in continuous:
        pair_type = "DC"
        logging.debug('"', U, '" and "', V, '" are data pair type', pair_type)
    else:
        pair_type = "Err"
        logging.warning("Error on", U, "and", V)

    return pair_type


def calc_mi(
        df,
        U,
        V,
        discrete,
        continuous,
        debug=False,
        charter="Plotly",
        chart=False,
        output=False,
        output_dir=None,
        resolution=150,
):
    """
    Takes two feature names and determines which mutual information method to use;
    returns calculated mutual information score
    """
    try:
        pairdf = df.filter([U, V]).dropna(how="any")

        if pairdf.shape[0] < 1:
            return 0

        logging.debug(
            "Calculating mutual information for",
            U,
            "(",
            list(df.columns).index(U),
            "of",
            len(list(df.columns)),
            ")",
            V,
            "(",
            list(df.columns).index(V),
            "of",
            len(list(df.columns)),
            ")",
        )

        mi_start_time = datetime.now()
        if U == V:
            return 1
        else:
            pair_type = calc_pairtype(U, V, discrete, continuous, debug=debug)
            # If both features are discrete:
            if pair_type == "DD":
                mi = DD_mi(pairdf, debug=debug)
            # If both features are continuous:
            elif pair_type == "CC":
                mi = CC_mi(pairdf, debug=debug)
            # If one feature is continuous and one feature is discrete:
            elif pair_type == "DC":
                mi = DC_mi(pairdf, continuous, debug=debug)
            else:
                mi = 0

            if chart:
                viz(
                    U,
                    V,
                    df,
                    discrete,
                    continuous,
                    charter=charter,
                    chart=chart,
                    output=output,
                    output_dir=output_dir,
                    resolution=resolution,
                )

            logging.debug("Elapsed time:", datetime.now() - mi_start_time)

        return mi
    except:
        return 0


def run_calc(
        features,
        df,
        discrete,
        continuous,
        debug=False,
        charter="Plotly",
        chart=False,
        output=False,
        output_dir=None,
        resolution=150,
):
    """
    Calculate the mutual information between every pair of columns in df.
    Return a dataframe whose columns are 'x' (the first column in the pair), 'y' (the second column),
    and 'v' (the mutual information of the pair). Each row is a distinct pair of variables and the MI between
    them.
    """
    start_time = datetime.now()
    pairs = list(itertools.combinations(df.columns, 2))
    xs, ys = zip(*pairs)
    vs = [
        calc_mi(
            df,
            x,
            y,
            discrete,
            continuous,
            debug=debug,
            charter=charter,
            chart=chart,
            output=output,
            output_dir=output_dir,
            resolution=resolution,
        )
        for x, y in pairs
    ]
    logging.debug("Elapsed time:", datetime.now() - start_time)
    logging.debug(
        "Calcuated mutual information for",
        len(features),
        "columns across",
        df.shape[0],
        "records",
    )

    return pd.DataFrame({"x": xs, "y": ys, "v": vs})


def calculate_positions(G):
    # Generate position data for each node
    pos = nx.kamada_kawai_layout(G, weight="weight")

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
    nodes = dict(
        x=node_x,
        y=node_y,
        name=node_name,
        adjacencies=node_adjacencies,
        centralities=node_centralities,
    )

    return edges, nodes


def draw_graph(
        edges,
        nodes,
        title,
        chart=False,
        output=False,
        output_dir=None,
        resolution=150,
        **kwargs,
):
    # Draw edges
    edge_trace = go.Scatter(
        x=edges["x"],
        y=edges["y"],
        line=dict(width=0.5, color="#888"),
        mode="lines+markers",
        hoverinfo="text",
    )

    # Draw nodes
    node_trace = go.Scatter(
        x=nodes["x"],
        y=nodes["y"],
        # Optional: Add labels to points *without* hovering (can get a little messy)
        mode="markers+text",
        # ...or, just add markers (no text)
        # mode='markers',
        text=nodes["name"],
        hoverinfo="text",
    )
    filename = title.lower().replace(" ", "_")

    # Color the node by its number of connections
    # node_trace.marker.color = nodes['adjacencies']
    node_trace.marker.color = nodes["centralities"]

    # Draw figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=120),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
        ),
    )

    fig.update_traces(textposition="top center")
    # Show figure
    if chart:
        fig.show()

    if output:
        fig.write_image(output_dir / "graph_example.png", scale=resolution // 72)


def find_max_component_threshold(stack):
    """
    Find the mutual information threshold that maximizes the number of connected components (subgraphs with >1 node)
    in the resulting graph.
    :param stack: every row is a pair of features and their associated mutual information.
    :return: the mutual information threshold.
    """

    # Create a data frame of edge counts and number of components for a given threshold
    e = pd.DataFrame(columns=["mi_threshold", "edge_count", "components"])

    # Fill in the 'e' data frame with the number of edges and number of components across a range of thresholds
    for i in np.arange(
            np.round(stack["v"].min(), 2), np.round(stack["v"].max(), 2), 0.01
    ):
        s = stack[stack["v"] > i]

        G = nx.Graph()
        G.add_nodes_from(
            list(dict.fromkeys((list(s["x"].unique()) + list(s["y"].unique()))))
        )
        G.add_edges_from(list(zip(s["x"], s["y"])))

        e = e.append(
            {
                "mi_threshold": i,
                "edge_count": (stack["v"] > i).sum(),
                "components": nx.number_connected_components(G),
            },
            ignore_index=True,
        )

    # Plot the number of edges for a range of mutual information scores
    sns.lineplot(e["mi_threshold"], e["edge_count"])

    # Plot the number of components for a range of mutual information scores
    sns.lineplot(e["mi_threshold"], e["components"])

    # Find the mutual information threshold which maximizes the component count
    max_component_threshold = e[e["components"] == max(e["components"])].max()[
        "mi_threshold"
    ]
    # optional: if there are multiple MI thresholds which maximize the number of components,
    # you may want to experiment with thresholding at .min()['mi_threshold'] instead of .max()['mi_threshold']
    # depending on your application; we have selected the maximum threshold for maximizing component counts,
    # as this minimizes the edge count over the minimum threshold, which reduces the visualization exploration space,
    # while still maximizing component counts

    return max_component_threshold


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

        if "_id" in i:
            ignore.append(i)

    df = df.drop(columns=ignore)
    df = df.replace(np.nan, None)
    df = df.replace("nan", None)
    df.columns = [c.replace(" ", "_") for c in df.columns]
    logging.debug(
        f"Loaded data from {input_file} with {df.shape[0]} observations and {df.shape[1]} features"
    )

    return df


def classify_features(df, discrete_threshold, debug=False):
    ## Identify feature type

    # Get a list of all response types
    response_list = pd.DataFrame(columns=["responses", "types"], index=df.columns)
    response_list["responses"] = [list(df[col].unique()) for col in df.columns]
    response_list["response_count"] = response_list["responses"].map(len)
    # Delete columns from the dataframe that only have one response
    response_list["only_one_r"] = [(len(r) < 2) for r in response_list["responses"]]
    only_one_r = list(response_list[response_list["only_one_r"] == True].index)
    df = df.drop(columns=only_one_r)
    response_list = response_list.drop(index=only_one_r)

    response_list["types"] = [get_types(col, response_list) for col in df.columns]
    response_list["string"] = [t["strings"] > 0 for t in response_list["types"]]
    response_list["float"] = [t["floats"] > 0 for t in response_list["types"]]

    # Classify features as discrete or continuous
    response_list["class"] = [
        "d" if ((len(r) < discrete_threshold) or (t["strings"] > 0)) else "c"
        for r, t in zip(response_list["responses"], response_list["types"])
    ]

    # Store these groups in a list
    discrete = list(response_list[response_list["class"] == "d"].index)
    continuous = list(response_list[response_list["class"] == "c"].index)
    response_counts = {
        feature: response_list["response_count"][feature]
        for feature in response_list.index
    }

    logging.debug(
        f"Counted {len(discrete)} discrete features and {len(continuous)} continuous features"
    )

    return discrete, continuous, response_counts


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sirius Data Processing Pipeline")
    parser.add_argument(
        "--dpi", type=int, default=150, help="resolution of output plots"
    )
    parser.add_argument(
        "--discrete-threshold",
        type=int,
        default=5,
        help="Number of responses below which numeric features are considered discrete",
    )
    parser.add_argument(
        "--chart",
        action="store_true",
        default=False,
        help="Display images while running computation",
    )
    parser.add_argument(
        "--charter",
        choices=["Plotly", "Seaborn"],
        default="Plotly",
        help="The plotting library to use.",
    )
    # TODO: Can probably just replace with the more common --verbose flag
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print updates to the console while running.",
    )
    parser.add_argument(
        "--output",
        action="store_true",
        default=False,
        help="Output json and pngs to files.",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        default=False,
        help="Do not output pair plots, network graph, or json.",
    )
    parser.add_argument(
        "--no-mi",
        action="store_true",
        default=False,
        help="Do not compute MI. Use cached MI values instead.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        default=False,
        help="Cache MI values to use later when generating visualizations.",
    )
    parser.add_argument(
        "--sample-n",
        default=None,
        type=int,
        help="Subsample the data. By default, work with all the data.",
    )
    parser.add_argument(
        "--input-file",
        default="example_data/data.csv",
        help="Location of the input CSV data.",
    )
    parser.add_argument(
        "--output-dir",
        default="example_data/output",
        help="A directory in which to store the json and png files.",
    )
    args = parser.parse_args()

    # Parameter settings
    # boolean for whether to display images while running computation
    chart = args.chart
    # boolean for whether to print updates to the console while running
    debug = args.debug
    output = args.output  # boolean for whether to output json and pngs to files
    cache = args.cache
    no_mi = args.no_mi
    no_viz = args.no_viz
    charter = args.charter  # accepts 'Seaborn' or 'Plotly'
    resolution = args.dpi  # int for resolution of output plots
    # number of responses below which numeric responses are considered discrete
    discrete_threshold = args.discrete_threshold
    sample_n = args.sample_n  # Work with all data (None), or just a sample?
    input_file = Path(args.input_file)  # e.g. './example_data/data.csv'
    output_dir = Path(args.output_dir)  # e.g. './example_data/output'
    # cd = 'example_data/output'

    # Configure logging
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    sns.set_style("whitegrid")

    # create output directories, if needed
    if cache:
        output_dir.mkdir(parents=True, exist_ok=True)

    if output:
        for d in ["charts", "json"]:
            (output_dir / d).mkdir(parents=True, exist_ok=True)

    df = load_data(input_file, sample_n=sample_n, debug=debug)

    discrete, continuous, response_counts = classify_features(
        df,
        discrete_threshold,
        debug=debug,
    )
    # drop features with only a single response
    df = df.drop(columns=[col for col in response_counts if response_counts[col] <= 1])
    # Format values of discrete columns as strings and continuous columns as floats
    for col in df.columns:
        if col in discrete:
            df[col] = df[col].map(str)
        elif col in continuous:
            df[col] = df[col].map(float)
        else:
            logging.warning("Error formatting column ", col)

    if not no_mi:
        # Calculation
        stack = run_calc(
            list(df.columns),
            df,
            discrete,
            continuous,
            debug=debug,
            charter=charter,
            chart=chart,
            output=output,
            output_dir=output_dir,
            resolution=resolution,
        )

        if cache:
            stack.to_csv(output_dir / "results.csv", index=False)

    # Network Graphing

    if no_viz:
        return

    if cache:
        # Re-import the Mutual Information results
        # (this is helpful if you want to re-generate visualizations
        # without having to re-run the mutual information calculations)
        stack = pd.read_csv(output_dir / "results.csv")

    # Sort our values and (optionally) exclude Mutual Infomation scores above 1 (which are often proxies for one another)
    sorted_stack = stack.sort_values(by="v", ascending=False)
    # sorted_stack = sorted_stack[sorted_stack['v'] < 1]

    # Thresholding

    max_component_threshold = find_max_component_threshold(sorted_stack)

    # Threshold the edge list by the mutual information threshold which maximizes the component count
    thresh_stack = sorted_stack[sorted_stack["v"] > max_component_threshold]
    thresh_stack = thresh_stack.rename(
        columns={"x": "source", "y": "target", "v": "weight"}
    )
    thresh_stack["viztype"] = [
        calc_pairtype(x, y, discrete, continuous)
        for x, y in zip(thresh_stack["source"], thresh_stack["target"])
    ]

    # Node and Edge Lists

    # Create a networkx graph from the list of pairs
    G = nx.from_pandas_edgelist(thresh_stack, "source", "target", ["weight"])

    nodelist = []
    for n in set(thresh_stack["source"]).union(thresh_stack["target"]):
        nodelist.append(
            {
                "name": n,
                "type": "continuous" if n in continuous else "discrete",
                "neighbors": list(dict(G[n]).keys()),
            }
        )

    json_out = {}
    json_out["nodes"] = nodelist
    json_out["links"] = thresh_stack.to_dict(orient="records")
    # json_out['edges'] = (thresh_stack).to_dict(orient='records')

    with open(output_dir / "graph.json", "w") as json_file:
        json.dump(json_out, json_file)

    for i, row in thresh_stack.iterrows():
        viz(
            row["source"],
            row["target"],
            df,
            discrete,
            continuous,
            charter=charter,
            chart=chart,
            output=output,
            output_dir=output_dir,
            resolution=resolution,
        )
        # viz(row['src'], row['target'], df, discrete, continuous)

    # Positioning
    [edges, nodes] = calculate_positions(G)

    draw_graph(
        edges,
        nodes,
        "Example Graph",
        chart=chart,
        output=output,
        output_dir=output_dir,
        resolution=resolution,
    )

    components = []
    for i in nx.connected_components(G):
        components.append(list(i))

    if output:
        with open(output_dir / "components.json", "w") as json_file:
            json.dump(components, json_file)


if __name__ == "__main__":
    import cProfile

    cProfile.run("main()")
    # main()
