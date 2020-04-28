import numpy as np
import pandas as pd
from datetime import datetime
import itertools
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score


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
    df = df.drop(columns=unary_cols, errors='ignore')
    feature_info = feature_info.drop(index=unary_cols, errors='ignore')

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
