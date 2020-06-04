import axios from "axios";
import {saveGraph, saveFilterOptions} from "./store";

export function loadGraphJSON(url:string, store:any):Promise<any> {
    return axios.get(url)
        .then((response) => {
          return response.data;
        })
        .then((data) => {
            store.dispatch(saveFilterOptions(getNodeProperties(data)));
            store.dispatch(saveGraph(convertToCYFormat(data)));
        })
}

/**
 * Get properties of the nodes frm the graph for use in the filter panel.
 * @param graphDictionary Json representation of the graph loaded from the server.
 */
function getNodeProperties(graphDictionary:any) {

    let nodeIds:Array<string> = [];
    let nodeTypes:Set<string> = new Set()
    let minWeight = Number.MAX_VALUE;
    let maxWeight = Number.MIN_VALUE;

    let nodeList = graphDictionary['nodes']
    for (let i = 0; i < nodeList.length; i++) {
        let node = nodeList[i];
        nodeIds.push(node.name);
        nodeTypes.add(node.type);
    }

    let edgeList = graphDictionary['links']
    for (let j = 0; j < edgeList.length; j++) {
        let edge = edgeList[j];
        minWeight = Math.min(minWeight, edge.weight);
        maxWeight = Math.max(maxWeight, edge.weight);
    }

    nodeIds.sort();

    return {
        nodeIds,
        nodeTypes: Array.from(nodeTypes),
        edgeWeightExtents: [minWeight, maxWeight]
    }
}

function convertToCYFormat(graphDictionary:any):any {
    return {
        'nodes': graphDictionary['nodes'].map(
                function (n):any {
                    return {
                        data: {
                            id: n.name,
                            type: n.type,
                            neighbors: n.neighbors
                        }
                    };
                }
            ),
        'edges': graphDictionary['links'].map(
            function (e:any):any {
                return {
                    data: {
                        id: e.source + '-' + e.target,
                        source: e.source,
                        target: e.target,
                        weight: e.weight,
                        viztype: e.viztype
                    }
                };
            }
        )
    };

}

// const elements = [
//    { data: { id: 'one', label: 'Node 1' }, position: { x: 0, y: 0 } },
//    { data: { id: 'two', label: 'Node 2' }, position: { x: 100, y: 0 } },
//    { data: { source: 'one', target: 'two', label: 'Edge from Node1 to Node2' } }
// ];


/**
 * Tries to load the JSON by alternating node names.
 * It is not clear which name should be first from the graph.
 * @param {string} sourceName
 * @param {string} targetName
 */
export function loadChartJson(sourceName:string, targetName:string) {
    // @ts-ignore chartJsonPath in the global scope. Check the html.
    let uploadFolderPath = chartJsonPath;
    return axios.get(uploadFolderPath + sourceName + "_" + targetName + ".json" )
        .catch(() => {
            return axios.get(uploadFolderPath + targetName + "_" + sourceName + ".json")
        })
        .then((response:any) => {
            return response.data
        })
}