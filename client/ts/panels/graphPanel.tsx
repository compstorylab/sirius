import * as React from 'react'
import * as ReactDOM from "react-dom";

import { connect } from 'react-redux'

import cytoscape from 'cytoscape/dist/cytoscape.esm.js';

export interface GraphPanelProps  {
    id?:string,
    className?:string,
    style?:any,
    graphData?:any
}

const defaultConfig = {
    style: [
        {
            selector: 'node',
            style: {
                'label': 'data(id)',
                'background-color': 'white',
                "width": "10px",
                "height": "10px",
                "font-size": "6px",
                "color": "#fff",
            }
        },
        {
            selector: "edge",
            "style": {
                "opacity": "0.4",
                "line-color": "white",
                "width": "2",
                "overlay-padding": "3px"
            }
        },
    ]
}

class GraphPanel extends React.Component<GraphPanelProps> {
    _cy:any;

    constructor(props) {
        super(props);
    }

    componentDidMount() {
        const container = ReactDOM.findDOMNode(this);
        this._cy = new cytoscape(Object.assign({}, defaultConfig, {container: container}));
        this._cy.on(
            'click',
            'edge',
            (evt:any) => {
                var node = evt.target;
                console.log(node.data() );
            }
        )

         const elements = this.props.graphData;
         if (elements) {
            this.updateCytoscape(null, this.props);
         }
    }

    updateCytoscape(prevProps, newProps) {
        const cy = this._cy;
        const { graphData } = newProps;
        // patch(cy, prevProps, newProps, diff, toJson, get, forEach);

        this._cy.json({ elements: graphData });
        this._cy.layout({name: 'cose'}).run();

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