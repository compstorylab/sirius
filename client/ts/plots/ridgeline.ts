/**
 * process Discrete-Continuous data into ridgeline chart ready data
 * @param data: {x: [list of string, discrete], y:[list of float, continuous], attributes:[x_name, y_name]}
 * return {x: [list of UNIQUE string, discrete], y:{ unique_x_1: [list of float], unique_x_2:[list of float]}, attributes:[x_name, y_name]}
 */
function processDCData(data, sourceName:string, sourceType:string, targetName:string, targetType:string){
    let discreteName:string = '';
    let continuousName:string = '';
    if(sourceType == 'discrete' && targetType == 'continuous') {
        discreteName = sourceName;
        continuousName = targetName;
    }
    else {
        discreteName = targetName;
        continuousName = sourceName;
    }

    let uniqueX:Set<any> = new Set();
    let indexedLists = {};

    let discreteData = data[discreteName]
    let continuousData = data[continuousName]
    for (const property in discreteData) {
        uniqueX.add(discreteData[property])

        if (!indexedLists.hasOwnProperty(discreteData[property])){
            indexedLists[discreteData[property]] = [];
        }
        indexedLists[discreteData[property]].push(continuousData[property]);
    }

    return {
        x: Array.from(uniqueX),
        y: indexedLists,
        xName: discreteName,
        yName: continuousName
    }
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
        opacity: 0.8,
        orientation: "h",
        side: 'positive',
        points: false,
        box: {visible: false},
        meanline: {visible: false},

    };
}

/**
 * create ridgeline chart using positive side violine chart
 * @param data
 * @param sourceName
 * @param sourceType
 * @param targetName
 * @param targetType
 */
// TODO: the json structure can be updated to avoid the expensive processDCData() process
export function ridgelineChart(data:any, sourceName:string, sourceType:string, targetName:string, targetType:string): any {
    let readyData = processDCData(data, sourceName, sourceType, targetName, targetType);
    let chartData = readyData.x.map((d: any) => {
        return prepareTrace(readyData.y[d], d)
    });

    let layout = {
        // title: 'hey new graph', // need configuration
        legend: {tracegroupap:0},
        xaxis: {showgrid: false, zeroline: false, title: readyData.yName},
        yaxis: {title: readyData.xName},
        font: {color: 'black'},

    };
    return {chartData, layout};
}