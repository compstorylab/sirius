import * as React from 'react'
import * as ReactDOM from "react-dom";

import { connect } from 'react-redux'

import cytoscape from 'cytoscape/dist/cytoscape.esm.js';
import cola from 'cytoscape-cola/cytoscape-cola.js';
console.log(cytoscape)
console.log(cola)

export interface GraphPanelProps  {
    id?:string,
    className?:string,
    style?:any,
    graphData?:any
}

let defaults = {
  name: 'euler',
  springLength: edge => 80,
  springCoeff: edge => 0.0008,
  mass: node => 4,
  gravity: -1.2,
  pull: 0.001,
  theta: 0.666,
  dragCoeff: 0.02,
  movementThreshold: 1,
  timeStep: 20,
  refresh: 10,
  animate: true,
  animationDuration: undefined,
  animationEasing: undefined,
  maxIterations: 500,
  maxSimulationTime: 4000,
  ungrabifyWhileSimulating: false,
  fit: true,
  padding: 30,
  boundingBox: undefined,
  ready: function(){}, // on layoutready
  stop: function(){}, // on layoutstop
  randomize: false
};


class GraphPanel extends React.Component<GraphPanelProps> {
    _cy:any;

    constructor(props) {
        super(props);
    }

    componentDidMount() {
        const container = ReactDOM.findDOMNode(this);
        this._cy = new cytoscape({
            container,
        });

         const elements = this.props.graphData;
         if (elements) {
            // console.log(this._cy.json())
            this.updateCytoscape(null, this.props);
         }
    }

    updateCytoscape(prevProps, newProps) {
        const cy = this._cy;
        const { graphData } = newProps;
        // patch(cy, prevProps, newProps, diff, toJson, get, forEach);

        this._cy.json({ elements: graphData });
        this._cy.layout({name: 'breadthfirst'}).run();

    }

    componentDidUpdate(prevProps) {
        this.updateCytoscape(prevProps, this.props);
    }

    componentWillUnmount() {
        this._cy.destroy();
    }

    render() {
        const { id, className, style } = this.props;
        return <div id={id} className={className} style={{ width: '100%', height: '900px' }}></div>
    }
}


const mapStateToProps = (state /*, ownProps*/) => {
  return {
    graphData: state.graphData
  }
}

const mapDispatchToProps = {}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(GraphPanel)