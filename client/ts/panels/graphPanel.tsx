import * as React from 'react'
import * as ReactDOM from "react-dom";
import { connect } from 'react-redux'
import cytoscape from 'cytoscape/dist/cytoscape.esm.js';

import {loadChartJson} from '../data'
import {graphStyle} from './graphStyle'
import {saveChartInfo} from '../store'

export interface GraphPanelProps  {
    id?:string,
    className?:string,
    style?:any,
    graphData?:any,
    saveChartInfo:any,
    edgeWeightExtents:any
}


class GraphPanel extends React.Component<GraphPanelProps> {
    _cy:any;

    constructor(props) {
        super(props);
    }

    onEdgeClick = (evt:any) => {
        let edge = evt.target;
        let chartInfo:any = edge.data()
        let source:string = chartInfo.source;
        let target:string = chartInfo.target;
        let vizType:string = chartInfo.viztype;

        let sourceType = edge.source().data().type
        let targetType = edge.target().data().type
        loadChartJson(source, target)
            .then((data) => {
                this.props.saveChartInfo({
                    source,
                    sourceType,
                    target,
                    targetType,
                    vizType,
                    data: data
                });
            });
    }

    onNodeSelect = (evt:any) => {
        var node = evt.target;

        this._cy.nodes().unselect();
        this._cy.nodes().removeClass('highlighted');
        this._cy.nodes().removeClass('selected');
        this._cy.edges().removeClass('highlighted');

        node.openNeighborhood().addClass('highlighted');
        node.addClass('selected');
    }

    componentDidMount() {
        const container = ReactDOM.findDOMNode(this);
        const defaultConfig = {
            style: graphStyle()
        }
        this._cy = new cytoscape(Object.assign({}, defaultConfig, {container: container}));
        this._cy.on('click', 'edge', this.onEdgeClick);
        this._cy.on('click', 'node', this.onNodeSelect);

         const elements = this.props.graphData;
         if (elements) {
            this.updateCytoscape(null, this.props);
         }
    }

    updateCytoscape(prevProps, newProps) {
        const cy = this._cy;
        const { graphData, filterSelections } = newProps;

        if(!prevProps.graphData) {
            this._cy.json({ elements: graphData });
            const layoutConfig =  {
                name: 'cose',
                animate: false
            };
            this._cy.layout(layoutConfig).run();
        }

        this._cy.nodes().removeClass('highlighted');
        this._cy.nodes().removeClass('selected');
        this._cy.edges().removeClass('highlighted');
        if (filterSelections && filterSelections.nodeNames) {
            let t = this._cy.nodes().filter((ele:any) => {
                return filterSelections.nodeNames.indexOf(ele.id()) > -1
            });

            t.openNeighborhood().addClass('highlighted');
            t.removeClass('highlighted');
            t.addClass('selected');


            this._cy.fit(t, 250)
        }
    }

    componentDidUpdate(prevProps) {
        this.updateCytoscape(prevProps, this.props);
    }

    componentWillUnmount() {
        this._cy.removeListener('click')
        this._cy.destroy();
    }

    render() {
        const { id, className, style } = this.props;
        return <div id={id} className={className} style={{ width: '100%', height: '900px' }}/>
    }
}


const mapStateToProps = (state /*, ownProps*/) => {
    return {
        graphData: state.graphData,
        filterSelections: state.filterSelections,
        edgeWeightExtents: state.filterOptions.edgeWeightExtents
    }
}

const mapDispatchToProps = {saveChartInfo}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(GraphPanel)