import Plotly from 'plotly.js-dist';

export function heatmap(x:Array<any>, y:Array<any>, chartHolderId:string): void {
    let data = processForHeatmap(x, y);
    var chartData = [
        {
            x: data['x'],
            y: data['y'],
            z: data['z'],
            type: 'heatmap',
            hoverongaps: false
        }
    ];
    Plotly.newPlot(chartHolderId, chartData);
}

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

function create2dZeroMatrix(x:number, y:number):Array<Array<number>> {
    let matrix = Array<Array<number>>();
    for(let i = 0; i < y; i++){
        let arr = new Array();
        for (let j = 0; j < x; j++) {
            arr.push(0);
        }
        matrix.push(arr);
    }
    return matrix;
}

function unique(arr):Array<any> {
    let unique_vals:Array<any> = Array.from(new Set(arr));
    unique_vals.sort();
    return unique_vals;
}

function getLabelIndexDict(labels){
    let d = {};
    for (let i = 0; i < labels.length; i++) {
        d[labels[i]] = i;
    }
    return d
}

