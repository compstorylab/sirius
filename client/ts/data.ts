import axios from "axios";

export function loadGraphJSON(url:string):Promise<any> {
    return axios.get(url)
       .then((response) => {
          return convertToCYFormat(response.data);
       })
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
