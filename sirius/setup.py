import argparse
import json
from pathlib import Path


DEFAULTS = {
    "dpi": 150,
    "discrete_threshold": 5,
    "debug": True,
    "output_json": True,
    "output_chart": True,
    "charter": "Plotly",
    "cache": True,
    "sample_n": None,
    "input_file": None,
    "output_dir": None,
    "output_limit_n": None,
}

def arg_setup():
    print('argsetup reached')
    parser = argparse.ArgumentParser(description='Sirius Data Processing Pipeline')
    params = DEFAULTS.copy()
    params_filename = './params.argv.json'
    if Path(params_filename).exists():
        with open(params_filename, 'r') as f:
            params.update(json.load(f))

    parser.add_argument('--dpi', type=int, default=params['dpi'], help='resolution of output plots')
    parser.add_argument('--discrete-threshold', type=int, default=params['discrete_threshold'],
                        help='Number of responses below which numeric features are considered discrete')
    parser.add_argument('--output-chart', action='store_true', default=params['output_chart'], help='Output charts to pngs')
    parser.add_argument('--charter', choices=['Plotly', 'Seaborn'], default=params['charter'], help='The plotting library to use.')
    parser.add_argument('--debug', action='store_true', default=params['debug'], help='Print updates to the console while running.')
    parser.add_argument('--output-json', action='store_true', default=params['output_json'], help='Output json to files.')
    parser.add_argument('--output-limit-n', type=int, default=params['output_limit_n'], help='Maximum number of data points to export into pairwise chart json files.')
    parser.add_argument('--cache', action='store_true', default=params['cache'], help='Cache MI values to use later when generating visualizations.')
    parser.add_argument('--sample-n', default=params['sample_n'], type=int, help='Subsample the data. By default, work with all the data.')
    parser.add_argument('--input-file', default=params['input_file'], help='Location of the input CSV data.')
    parser.add_argument('--output-dir', default=params['output_dir'], help='A directory in which to store the json and png files.')
    args, unknown = parser.parse_known_args()
    print('args:', vars(args))
    return vars(args)
