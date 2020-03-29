import Plotly from 'plotly.js-dist';
/**
 * pricess Discrete-Continuous data into ridgeline chart ready data
 * @param data: {x: [list of string, discrete], y:[list of float, continuous], attributes:[x_name, y_name]}
 * return {x: [list of UNIQUE string, discrete], y:{ unique_x_1: [list of float], unique_x_2:[list of float]}, attributes:[x_name, y_name]}
 */
function processDCData(data){
    let uniqueX = [... new Set(data.x)];
    let indexList = {};
    uniqueX.map(function(d: any){
        indexList[d] = []
    });
    data.x.map(function(d, i){
        indexList[d].push(data.y[i]);
    });
    let result = {
        x: uniqueX,
        y: indexList,
        attributes: data.attributes
    }
    return result
}
/**
 * prepare the trace data required to draw the ridgeline chart
 * @param d:  should be a list of float values
 * @param name: string, the value of one of the unique descrite x 
 */
function prepareTrace(d: Float32Array, name: String){
    return {
        type: 'violin',
        name: name,
        showlegend: false,
        x: d,
        width: 3,
        opacity: 0.5,
        orientation: "h",
        side: 'positive',
        points: false,
        box: {visible: false},
        meanline: {visible: false},

    };
}

/** 
 * create ridgeline chart using positive side violine chart
 * @param data: data read from json file
 * @param chartHolderId: string, indicate which DOM to insert the chart
 */
// TODO: the json structure can be updated to avoid the expensive processDCData() process
export function CreateRidgelineChart(data: any, chartHolderId: String): void {
    let readyData = processDCData(data),
        chartData = [];
    readyData.x.map((d: any)=>{
        chartData.push(prepareTrace(readyData.y[d], d))
    });

    var layout = {
        // title: 'hey new graph', // need configuration
        legend: {tracegroupap:0},
        xaxis: {showgrid: false, zeroline: false, title: readyData.attributes[1]}, 
        yaxis: {title: readyData.attributes[0]},
        plot_bgcolor: "rgba(0, 0, 0, 0)",
        paper_bgcolor: "rgba(0, 0, 0, 0)",
        font: {color: 'white'},

    };
    Plotly.newPlot(chartHolderId, chartData, layout);
}