from pathlib import Path

import pandas as pd

from . import setup
from .data_processing import classify_features, load_data
from .mutual_info import calc_mi
from .network import output_graph_json, output_pairs_json, threshold_using_backbone_method
from .visualization import draw_graph, viz


def main():
    
    args = setup.arg_setup()
    print("args['cache']:",args['cache'])

    # create output directories, if needed
    if args['cache']:
       Path(args['output_dir']).mkdir(parents=True, exist_ok=True)

    if args['output_json']:
        (Path(args['output_dir']) / 'json').mkdir(parents=True, exist_ok=True)

    if args['output_chart']:
        (Path(args['output_dir']) / 'charts').mkdir(parents=True, exist_ok=True)

    # Load Data
    df = load_data(Path(args['input_file']), sample_n=args['sample_n'], debug=args['debug'])

    # Classify Features
    feature_info = classify_features(df, args['discrete_threshold'], debug=args['debug'])
    if args['debug']:
        print('Classified Features:')
        print(feature_info)
        
    edges = calc_mi(df, feature_info, debug=args['debug'])
    if args['cache']:
        edges.to_csv(Path(args['output_dir']) / 'results.csv', index=False)
        # Re-import the Mutual Information results
        # (this is helpful if you want to re-generate visualizations
        # without having to re-run the mutual information calculations)
        edges = pd.read_csv(Path(args['output_dir']) / 'results.csv')
        

    # Sort our values and (optionally) exclude Mutual Infomation scores above 1 (which are often proxies for one another)
    edges = edges.sort_values(by='v', ascending=False)
    # sorted_stack = sorted_stack[sorted_stack['v'] < 1]

    # Threshold the edge list by the mutual information threshold which maximizes the component count
    thresheld_edges = threshold_using_backbone_method(edges)

    # Sirius JSON

    if args['output_json']:
        output_graph_json(thresheld_edges, feature_info, Path(args['output_dir']))
        output_pairs_json(df, Path(args['output_dir']), thresheld_edges, num_sample=args['output_limit_n'])

    # Visualizations

    if args['output_chart']:
        draw_graph(thresheld_edges, 'Example Graph', output_chart=args['output_chart'], output_dir=Path(args['output_dir']), resolution=args['dpi'])
        for x, y in zip(thresheld_edges['x'], thresheld_edges['y']):
            viz(x, y, df, feature_info, charter=args['charter'], output_chart=args['output_chart'], output_json=args['output_json'],
                output_dir=Path(args['output_dir']), resolution=args['dpi'])


if __name__ == '__main__':
    main()


