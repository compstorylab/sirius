import { createHeatMap } from "./heatmap";
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