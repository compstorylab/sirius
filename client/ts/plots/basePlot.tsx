import * as React from 'react'
import * as ReactDOM from "react-dom";

import Plotly from './plotlyApi';

import {heatmap} from './heatmap';
import {create2DDensityChart} from './twoDDensity';
import {ridgelineChart} from './ridgeline';

export interface PlotProps  {
    data:any,
    chartType:string,
    sourceName:string,
    targetName:string,
    sourceType:string,
    targetType:string
}

export class PlotlyPlot extends React.Component<PlotProps> {

    constructor(props:any) {
        super(props);
    }

    componentDidMount() {
        if(this.props.data) {
            switch (this.props.chartType) {
                case "DD":
                    this.plotHeatmap(this.props.data)
                    break;
                case "DC":
                case "CD":
                    this.plotRidgePlot(
                        this.props.data,
                        this.props.sourceName,
                        this.props.sourceType,
                        this.props.targetName,
                        this.props.targetType)
                    break;
                case "CC":
                    this.plot2DDensity(this.props.data)
                    break;
            }
        }
    }
    componentDidUpdate(prevProps) {

    }
    componentWillUnmount() {
        const container = ReactDOM.findDOMNode(this);
        Plotly.purge(container)
    }

    plotHeatmap(data:any){
        let keys = Object.keys(data);
        let xAxisTitle:string = keys[0];
        let yAxisTitle:string = keys[1];
        // @ts-ignore
        let xvals = Object.values(data[keys[0]]);
        // @ts-ignore
        let yvals = Object.values(data[keys[1]]);
        let {chartData, layout} = heatmap(xvals, yvals, xAxisTitle, yAxisTitle);
        const container = ReactDOM.findDOMNode(this);
        Plotly.react(container, chartData, layout)
    }

    plotRidgePlot(data:any, sourceName:string, sourceType:string, targetName:string, targetType:string){
        let {chartData, layout} = ridgelineChart(data, sourceName, sourceType, targetName, targetType);
        const container = ReactDOM.findDOMNode(this);
        Plotly.react(container, chartData, layout)
    }

    plot2DDensity(data:any){
        let {chartData, layout} = create2DDensityChart(data);
        const container = ReactDOM.findDOMNode(this);
        Plotly.react(container, chartData, layout)
    }

    render() {
        return <div/>
    }
}