from sirius import *

def main():
    
    args = setup.argSetup()
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
        
    stack = calc_mi(df, feature_info, debug=args['debug'])
    if args['cache']:
        stack.to_csv(Path(args['output_dir']) / 'results.csv', index=False)
        # Re-import the Mutual Information results
        # (this is helpful if you want to re-generate visualizations
        # without having to re-run the mutual information calculations)
        pair_info = pd.read_csv(Path(args['output_dir']) / 'results.csv')
    else:
        pair_info = stack
        

    # Sort our values and (optionally) exclude Mutual Infomation scores above 1 (which are often proxies for one another)
    pair_info = pair_info.sort_values(by='v', ascending=False)
    # sorted_stack = sorted_stack[sorted_stack['v'] < 1]

    # Threshold the edge list by the mutual information threshold which maximizes the component count
    pair_info = threshold_by_max_component(pair_info, output_chart=args['output_chart'], resolution=args['dpi'])

    # Sirius JSON

    if args['output_json']:
        output_graph_json(pair_info, feature_info, Path(args['output_dir']))
        output_pairs_json(df, Path(args['output_dir']), pair_info, num_sample=args['output_limit_n'])

    # Visualizations

    if args['output_chart']:
        draw_graph(pair_info, 'Example Graph', output_chart=args['output_chart'], output_dir=Path(args['output_dir']), resolution=args['dpi'])
        for x, y in zip(pair_info['x'], pair_info['y']):
            viz(x, y, df, feature_info, charter=args['charter'], output_chart=args['output_chart'], output_json=args['output_json'],
                output_dir=Path(args['output_dir']), resolution=args['dpi'])




if __name__ == '__main__':
    import cProfile
    cProfile.run('main()')



