import * as React from "react"
import { connect } from 'react-redux'

import Form from 'react-bootstrap/Form'

import {saveFilterSelections} from '../store'

export interface SidebarProps {
    filterOptions?:any,
    saveFilterSelections:any
}

/**
 * This is the filter panel that sits on the right side of the screen.
 * It is used to select nodes within the adjoining network graph.
 */
class Sidebar extends React.Component<SidebarProps> {

    /**
     * Gets all selections from fields in the filter panel
     */
    gatherFilterSelections(){
        let nodeNameSelector:HTMLSelectElement = document.getElementById('node-name-selector') as HTMLSelectElement
        let selectedOptions = nodeNameSelector.selectedOptions;
        let selectedNames = Array.from(selectedOptions)
                                  .map((x:HTMLOptionElement) => x.label)

        let nodeTypeCheckBoxes = document.querySelectorAll('#node-type-checkboxes input');
        let nodeTypeSelections = Array.from(nodeTypeCheckBoxes)
                                      .filter((x:HTMLInputElement) => x.checked)
                                      .map(function(x:HTMLOptionElement) {
                                          return x.id
                                      });
        return {
            nodeNames: selectedNames,
            nodeTypes: nodeTypeSelections
        }
    }

    /**
     * Event handler for name select input
     */
    onNodeNameSelectChange = (evt:any) => {
        this.props.saveFilterSelections(
            this.gatherFilterSelections())
    }

    /**
     * Handler for node type check boxes
     * Note: These have been commented out
     */
    onNodeTypeCheckBoxChange = (evt:any) => {
        this.props.saveFilterSelections(
            this.gatherFilterSelections())
    }

    render() {
        if (this.props.filterOptions === null) {
            return null
        }

        let nodeNames:Array<string> = this.props.filterOptions.nodeIds;
        if(!nodeNames || nodeNames.length === 0) {
            nodeNames = ['No Data'];
        }

        let nodeTypeOption:Array<any> = this.props.filterOptions.nodeTypes;

        return <div>
            <h3 className={"mt-4"} >Graph Filter</h3>
            <Form>
                <Form.Group>
                    <Form.Label>Feature names</Form.Label>
                    <Form.Control style={{'height': '200px'}}
                                  id={'node-name-selector'}
                                  onChange={this.onNodeNameSelectChange}
                                  as={"select"} multiple>
                    {nodeNames.map((nodeName, i:number) => {
                        return <option key={`_${i}`}>{nodeName}</option>
                    })
                    }
                    </Form.Control>
                </Form.Group>
                {/*<Form.Group>*/}
                {/*    <Form.Label>Feature type</Form.Label>*/}
                {/*    <div id={"node-type-checkboxes"}>*/}
                {/*        {nodeTypeOption.map((type, i) => (*/}
                {/*            <div key={`ntcb-${i}`} className="mb-3">*/}
                {/*              <Form.Check*/}
                {/*                type='checkbox'*/}
                {/*                id={type}*/}
                {/*                label={type}*/}
                {/*                onChange={this.onNodeTypeCheckBoxChange}*/}
                {/*                defaultChecked*/}
                {/*              />*/}
                {/*            </div>*/}
                {/*          ))}*/}
                {/*    </div>*/}
                {/*</Form.Group>*/}
            </Form>
        </div>
    }
}

const mapStateToProps = (state /*, ownProps*/) => {
  return {
    filterOptions: state.filterOptions
  }
}

const mapDispatchToProps = {saveFilterSelections}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(Sidebar)