import axios from 'axios';
import * as React from 'react'
import { connect } from 'react-redux'
import Modal from "react-bootstrap/Modal";

import {PlotlyPlot} from '../plots/basePlot';
import {saveChartInfo} from '../store'

export interface ChartOverlayProps {
    saveChartInfo:any,
    chartInfo?:any,
}

class ChartOverlay extends React.Component<ChartOverlayProps> {

    constructor(props:any) {
        super(props);
    }

    onClose = (evt:any) => {
        this.props.saveChartInfo(null);
    }

    render() {
        let title:string = '';
        let vizType:string = '';
        let chartData:any = null;
        let sourceName:string = '';
        let sourceType:string = '';
        let targetName:string = '';
        let targetType:string = '';
        if(this.props.chartInfo){
            title = `${this.props.chartInfo.source} vs ${this.props.chartInfo.target}`;
            vizType = this.props.chartInfo.vizType;
            chartData = this.props.chartInfo.data;
            sourceName = this.props.chartInfo.source;
            sourceType = this.props.chartInfo.sourceType;
            targetName = this.props.chartInfo.target;
            targetType = this.props.chartInfo.targetType;
        }

        return  (
            <Modal size="lg"
                   show={this.props.chartInfo !== null}
                   onHide={this.onClose}>
                <Modal.Header closeButton >
                    <Modal.Title id="contained-modal-title-vcenter">{title}</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    <PlotlyPlot data={chartData}
                                chartType={vizType}
                                sourceName={sourceName}
                                sourceType={sourceType}
                                targetName={targetName}
                                targetType={targetType}/>
                </Modal.Body>
            </Modal>
        );
    }
}

const mapStateToProps = (state /*, ownProps*/) => {
    return {
        chartInfo: state.chartInfo,
    }
}

const mapDispatchToProps = {saveChartInfo}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(ChartOverlay)
