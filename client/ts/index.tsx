import * as React from 'react';
import * as ReactDOM from "react-dom";

import {loadGraphJSON} from './data';
import Sidebar from './panels/sidebar';
import GraphPanel from './panels/graphPanel';
import ChartOverlay from './panels/chartOverlay';

import { Provider } from 'react-redux'
import { store } from './store'



let jsonUrl:HTMLInputElement = document.getElementById('json-url') as HTMLInputElement;
if (jsonUrl && jsonUrl.value){
   loadGraphJSON(jsonUrl.value, store)
       .then(() => {
           document.body.style.backgroundImage = 'None';
       });
}

ReactDOM.render(
    <Provider store={store}>
        <Sidebar/>
    </Provider>,
    document.getElementById('sidebar')
)
ReactDOM.render(
    <Provider store={store}>
        <GraphPanel/>
    </Provider> ,
    document.getElementById('network_graph')
)
ReactDOM.render(
    <Provider store={store}>
        <ChartOverlay/>
    </Provider> ,
    document.getElementById('chart_overlay')
)
