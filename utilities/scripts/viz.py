## Visualization

# Discrete-Discrete Confusion Matrices
def vizDD(df, charter='Plotly', chart=False, output=False, output_dir=None, resolution=150):
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
            with open(output_dir / 'charts' / U + '_' + V + '.json', 'w') as outfile:
                json.dump(fig.to_json(), outfile)

            fig.write_image(output_dir / 'charts' / U + '_' + V + '.png', scale=resolution // 72)
    else:
        plt.clf()
        plt.figure(dpi=resolution)
        sns.heatmap(s, annot=True, cmap="Blues", cbar=False, linewidths=1)
        plt.xlabel(U.replace('_', ' ').title())
        plt.ylabel(V.replace('_', ' ').title())
        if output:
            plt.savefig(output_dir / 'charts' / U + '_' + V + '.png', dpi=resolution)

        if chart:
            plt.show()

    plt.close('all')


# Discrete-Continuous Violin Plots
def vizDC(df, continuous, charter='Plotly', chart=False, output=False, output_dir=None, resolution=150):
    ''' Takes a subset dataframe of one continuous and one discrete feature and generates a Violin Plot '''

    U = list(df.columns)[0]
    V = list(df.columns)[1]

    if (U in continuous):
        D = V
        C = U
    else:
        D = U
        C = V

    if charter == 'Plotly':
        fig = go.Figure()
        for i in list(df[D].unique()):
            series = df[df[D] == i][C]
            fig.add_trace(go.Violin(x=series, name=str(i)))

        fig.update_traces(orientation='h', side='positive', width=3, points=False)
        fig.update_layout(
            xaxis_showgrid=False,
            xaxis_zeroline=False,
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
            with open(output_dir / 'charts' / U + '_' + V + '.json', 'w') as outfile:
                json.dump(fig.to_json(), outfile)

            fig.write_image(output_dir / 'charts' / U + '_' + V + '.png', scale=resolution // 72)
    else:
        sns.violinplot(df[D], df[C])
        if len(df[D]) < 500:
            sns.swarmplot(x=df[D], y=df[C], edgecolor="white",
                          linewidth=1)  # Only show a swarm plot if there are fewer than 500 data points
        plt.xlabel(D.replace('_', ' ').title())
        plt.ylabel(C.replace('_', ' ').title())

        if output:
            plt.savefig(output_dir / 'charts' / U + '_' + V + '.png', dpi=resolution)

        if chart:
            plt.show()

    plt.close('all')


# Continuous-Continuous KDE Plots
def vizCC(df, charter='Plotly', chart=False, output=False, output_dir=None, resolution=150):
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
            with open(output_dir / 'charts' / U + '_' + V + '.json', 'w') as outfile:
                json.dump(fig.to_json(), outfile)

            fig.write_image(output_dir / 'charts' / U + '_' + V + '.png', scale=resolution // 72)
    else:
        sns.kdeplot(df[U], df[V], color='blue', shade=True, alpha=0.3, shade_lowest=False)
        if len(df[U]) < 500:
            sns.scatterplot(x=df[U], y=df[V], color='blue', alpha=0.5,
                            linewidth=0)  # Only show a scatter plot if there are fewer than 500 data points

        plt.xlabel(U.replace('_', ' ').title())
        plt.ylabel(V.replace('_', ' ').title())
        if output:
            plt.savefig(output_dir / 'charts' / U + '_' + V + '.png', dpi=resolution)

        if chart:
            plt.show()

    plt.close('all')


# Matrix Heatmap
def vizHeatmap(matrix):
    plt.clf()
    plt.figure(dpi=70, figsize=(10, 8))
    sns.heatmap(matrix.fillna(0))
    plt.show()


# Visualization Function Router
def vizRouter(U, V, df, discrete, continuous, charter='Plotly', chart=False, output=False, output_dir=None, resolution=150):
    ''' Generate a visualization based on feature types '''
    plt.clf()
    plt.figure(dpi=resolution)

    pairdf = df.filter([U, V]).dropna(how='any')

    # If both features are discrete:
    if U in discrete and V in discrete:
        DD_viz(pairdf, charter=charter, chart=chart, output=output, output_dir=output_dir, resolution=resolution)
    # If both features are continuous:
    elif U in continuous and V in continuous:
        CC_viz(pairdf, charter=charter, chart=chart, output=output, output_dir=output_dir, resolution=resolution)
    # If one feature is continuous and one feature is discrete:
    elif U in continuous and V in discrete or U in discrete and V in continuous:
        DC_viz(pairdf, continuous, charter=charter, chart=chart, output=output, output_dir=output_dir, resolution=resolution)
    else:
        raise Exception('Error on features', U, 'and', V)

    if output:
        pairdf.to_json(output_dir / 'json' / U + '_' + V + '.json')

    return viz