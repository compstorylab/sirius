import math
import pandas as pd
import numpy as np


def load_data(input_file, sample_n=None, debug=False, ignore=None):
    if ignore is None:
        ignore = []
    # Load Data
    df = pd.read_csv(input_file)
    if sample_n:
        df = df.sample(sample_n)

    df = df.drop(columns=ignore)
    df = df.replace(np.nan, None)
    df = df.replace('nan', None)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    if debug:
        print(f'Loaded data from {input_file} with {df.shape[0]} observations and {df.shape[1]} features')

    return df


def compute_bandwidth(X, df):
    ''' Takes a column name and computes suggested gaussian bandwidth with the formula: 1.06*var(n^-0.2) '''
    var = np.var(df[X])
    n = len(df[X].notnull())
    b = 1.06 * var * (n ** (-0.2))
    return b


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
