import * as React from 'react'
import * as ReactDOM from "react-dom";

// import { types } from './types';
// import { defaults } from './defaults';
import cytoscape from 'cytoscape/dist/cytoscape.esm.js';
// import { patch } from './patch';


export interface GraphPanelProps  {
    id?:string,
    className?:string,
    style?:any
}

/**
 * The `CytoscapeComponent` is a React component that allows for the declarative creation
 * and modification of a Cytoscape instance, a graph visualisation.
 */
export class GraphPanel extends React.Component<GraphPanelProps> {
    _cy:any;

    constructor(props) {
        super(props);
    }

    componentDidMount() {
        const container = ReactDOM.findDOMNode(this);

         const elements = [
           { data: { id: 'one', label: 'Node 1' }, position: { x: 0, y: 0 } },
           { data: { id: 'two', label: 'Node 2' }, position: { x: 100, y: 0 } },
           { data: { source: 'one', target: 'two', label: 'Edge from Node1 to Node2' } }
        ];

         console.log(cytoscape)

        this._cy = new cytoscape({
            container,
            elements
        });
        this.updateCytoscape(null, this.props);
    }

    updateCytoscape(prevProps, newProps) {
        const cy = this._cy;
        const { diff, toJson, get, forEach } = newProps;
        // patch(cy, prevProps, newProps, diff, toJson, get, forEach);


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
