import * as React from "react"
import { connect } from 'react-redux'

import Form from 'react-bootstrap/Form'

import {saveFilterSelections} from '../store'

export interface SidebarProps {
    filterOptions?:any,
    saveFilterSelections:any
}

class Sidebar extends React.Component<SidebarProps> {

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

    onNodeNameSelectChange = (evt:any) => {
        this.props.saveFilterSelections(
            this.gatherFilterSelections())
    }

    onNodeTypeCheckBoxChange = (evt:any) => {
        this.props.saveFilterSelections(
            this.gatherFilterSelections())
    }

    render() {
        if (this.props.filterOptions === null) {
            return null
        }

        let nodeNames:Array<string> = this.props.filterOptions.nodeIds;
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
                    {nodeNames.map((nodeName) => {
                        return <option>{nodeName}</option>
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