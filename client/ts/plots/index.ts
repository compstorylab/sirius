import { createHeatMap } from "./heatmap";
import { CreateRidgelineChart } from "./ridgeline";
import { Create2DDensityChart } from "./twoDDensity";

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
            Create2DDensityChart(data, chartHolderId);
            break;
    }
}