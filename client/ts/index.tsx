import * as React from 'react'
import * as ReactDOM from "react-dom";

import {Sidebar} from './panels/sidebar'
import {GraphPanel} from './panels/graphPanel'

let jsonUrl:HTMLInputElement = document.getElementById('json-url') as HTMLInputElement;
if (jsonUrl && jsonUrl.value){
   console.log(jsonUrl.value)
}



ReactDOM.render(
    <Sidebar/>,
    document.getElementById('sidebar')
)
ReactDOM.render(
    <GraphPanel/>,
    document.getElementById('network_graph')
)


// import {Drawer} from "./drawer";
//
// import {generateGraphChart} from "./graph_chart";
//
// (function(){
//
//    let jsonUrl = <HTMLInputElement>document.getElementById('json-url'),
//        body = document.getElementsByTagName("body")[0],
//        uploadLink = document.getElementById("upload-link");
//
//    let drawer:Drawer = Drawer.getInstance();
//
//     uploadLink.addEventListener('click', () => {
//        drawer.open();
//     });
//
//    // draw graph network
//    if (jsonUrl && jsonUrl.value){
//       // remove background image, replace with background color
//       body.style.backgroundImage = 'None';
//       // hide right panel by default when graph chart is generated
//       drawer.close();
//
//       generateGraphChart(jsonUrl);
//    }
//
// }());
//
