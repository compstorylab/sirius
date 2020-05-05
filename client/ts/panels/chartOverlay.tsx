import * as React from 'react'
import * as ReactDOM from "react-dom";
import { connect } from 'react-redux'

import Col from "react-bootstrap/Col"
import Container from "react-bootstrap/Container"
import Modal from "react-bootstrap/Modal";
import Row from "react-bootstrap/Row"

import {saveChartInfo} from '../store'

export interface ChartOverlayProps {
    saveChartInfo:any,
    chartInfo?:any
}

class ChartOverlay extends React.Component<ChartOverlayProps> {

    onClose = (evt:any) => {
        console.log('close')
        this.props.saveChartInfo(null);
    }

    render() {
        if(this.props.chartInfo) {
            return  (
                <Modal size="lg"
                       show={true}
                       onHide={this.onClose}>
                    <Modal.Header closeButton >
                        <Modal.Title id="contained-modal-title-vcenter">Chart name</Modal.Title>
                    </Modal.Header>
                    <Modal.Body>
                        <Container>
                            <Row className="show-grid">
                                <Col xs={12} md={8}>
                                </Col>
                                <Col xs={6} md={4}>
                                </Col>
                            </Row>
                        </Container>
                    </Modal.Body>
                </Modal>
            );
        }
        else {
            return null
        }

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
