import Plotly from 'plotly.js-dist';

/**
 * Creates a heatmap from raw data using plotly.
 * @param x An array of categorical values. Number or string
 * @param y An array of categrorical values. Numnber or string
 * @param xName Title for the x-axis
 * @param yName Title for the y-axis
 * @param chartHolderId The id of the html element to render the chart in.
 */
export function heatmap(x:Array<any>, y:Array<any>, xName:string, yName:string, chartHolderId:string): void {
    let data = processForHeatmap(x, y);
    let chartData = [
        {
            x: data['x'],
            y: data['y'],
            z: data['z'],
            type: 'heatmap',
            hoverongaps: false
        }
    ];
    let layout = {
        xaxis: {
            title: {
                text: xName
            }
        },
        yaxis: {
            title: {
                text: yName
            }
        }
    };
    Plotly.newPlot(chartHolderId, chartData, layout);
}

/**
 * Transoform raw data into a form the heatmap function of Plotly can use. 
 * @param x An array of categorical values. Number or string
 * @param y An array of categrorical values. Numnber or string
 */
export function processForHeatmap(x:Array<any>, y:Array<any>) {
    let x_labels = unique(x);
    let y_labels = unique(y);

    let x_label_index = getLabelIndexDict(x_labels);
    let y_label_index = getLabelIndexDict(y_labels);

    let counts = create2dZeroMatrix(x_labels.length, y_labels.length);

    let count_max = -99999;
    let count_min = 99999;
    let n = x.length;
    for (let i = 0; i < n; i++){
        counts[y_label_index[y[i]]][x_label_index[x[i]]] ++
        count_max = Math.max(counts[y_label_index[y[i]]][x_label_index[x[i]]], count_max)
        count_min = Math.min(counts[y_label_index[y[i]]][x_label_index[x[i]]], count_min)
    }

    return {
        x: x_labels,
        y: y_labels,
        z: counts
    };
}

/**
 * Create a 2D array populated with zeros or a specified initial value.
 * @param x An integer denoting the number of columns
 * @param y An integer denoting the number of rows
 * @param initialValue The initial value.
 */
function create2dZeroMatrix(x:number, y:number, initialValue:any=0):Array<Array<number>> {
    let matrix = Array<Array<number>>();
    for(let i = 0; i < y; i++){
        let arr = new Array();
        for (let j = 0; j < x; j++) {
            arr.push(initialValue);
        }
        matrix.push(arr);
    }
    return matrix;
}

/**
 * Gets the unique values within an array.
 * @param arr An array on categorical values
 */
function unique(arr):Array<any> {
    let unique_vals:Array<any> = Array.from(new Set(arr));
    unique_vals.sort();
    return unique_vals;
}

/**
 * Creates a map from label value to array index.
 * getLabelIndexDict(['a', 'b']) -> {a:0, b:1}
 * @param labels An array of labels
 */
function getLabelIndexDict(labels){
    let d = {};
    for (let i = 0; i < labels.length; i++) {
        d[labels[i]] = i;
    }
    return d
}

