import { heatmap } from "./heatmap";
import { CreateRidgelineChart } from "./ridgeline";

/**
 * Selects plot create function by plot type
 * @param {string} type
 * @param data
 */
export function createPlot(type:string, data:any, chartHolderId:string) {
    switch(type) {
        case "DD":
            createHeatMap(data, chartHolderId);
            break;
        case "DC":
        case "CD":
            CreateRidgelineChart(data, chartHolderId);
            break;
        case "CC":
            // TODO: Add plot function
            break;
    }
}

function createHeatMap(data:any, chartHolderId:string) {
    let keys = Object.keys(data);
    let xAxisTitle:string = keys[0];
    let yAxisTitle:string = keys[1];
    let xvals = Object.values(data[keys[0]]);
    let yvals = Object.values(data[keys[1]]);
    heatmap(xvals, yvals, xAxisTitle, yAxisTitle,chartHolderId);
}