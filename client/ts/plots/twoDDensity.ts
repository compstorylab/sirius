/**
 * process Continuous-Continuous data into format that is ready for creating 2d density plot with histogram subplots
 * @param data read from raw json file, in the format of 
 * {
 *      feature_1_name: {0: value, 1:value, 2:value, ...},
 *      feature_2_name: {0: value, 1:value, 2:value, ...}
 * }
 * return an object in format: {x: [list of feature_1 value], y:[list of feature_2 value], attributes: [feature_1_name, feature_2_name]}
 */
function processCCData(data) {
    let x,
        y,
        attr_array = [];

    attr_array = Object.keys(data);
    // @ts-ignore
     x = Object.values(data[attr_array[0]])
    // @ts-ignore
    y = Object.values(data[attr_array[1]])

    return { "x": x, "y": y, "attributes": attr_array }
}

/**
 * create 2d density plot with histograms
 * @param data data read from json file
 */
export function create2DDensityChart(data:any):any {
    let readyData = processCCData(data);
    let x = readyData.x,
        y = readyData.y,
        feature_1_name = readyData.attributes[0],
        feature_2_name = readyData.attributes[1];

    
    let scatter_trace = {
        x: x,
        y: y,
        mode: 'markers',
        name: 'points',
        marker: {
          color: 'rgb(135, 206, 250)',
          size: 2,
          opacity: 0.8
        },
        type: 'scattergl'
    },
    density_trace = {
        x: x,
        y: y,
        name: 'density',
        colorscale: 'Blues',
        //ncontours: 20,
        reversescale: true,
        showscale: false,
        type: 'histogram2dcontour'
    },
    x_histogram = {
        x: x,
        name: 'x density',
        marker: {color: 'rgb(135, 206, 250)'},
        yaxis: 'y2',
        type: 'histogram'
    },
    y_histogram = {
        y: y,
        name: 'y density',
        marker: {color: 'rgb(135, 206, 250)'},
        xaxis: 'x2',
        type: 'histogram'
    };
    let chartData = [scatter_trace, density_trace, x_histogram, y_histogram],
        layout = {
            xaxis: { domain: [0, 0.85], showgrid: false, zeroline: false, title: feature_1_name },
            yaxis: { domain: [0, 0.85], showgrid: false, zeroline: false, title: feature_2_name },
            xaxis2: { domain: [0.85, 1],showgrid: false, zeroline: false },
            yaxis2: { domain: [0.85, 1], showgrid: false, zeroline: false },
            showlegend: false,
            hovermode: 'closest',
            bargap: 0,
            font: { color: 'black' },
        };
    return {
        chartData,
        layout
    };
}