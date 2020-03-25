import Plotly from 'plotly.js-dist';

export function ridgelineChart(x: Array<any>, y: Array<any>, xName: string, yName: string, chartHolderId: string): void {
    let chartData = [
        {
            type: 'violin',
            x: x,
            y: y,
            side: 'positive',
            x0: xName,
            y0: yName
        }
    ]
}