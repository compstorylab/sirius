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


// For file uploads
let fileSelectComponent = document.getElementById("upload-box");
if(fileSelectComponent) {
    fileSelectComponent.addEventListener('change', (event) => {
            let fileInput: HTMLInputElement = event.target as HTMLInputElement;
            let filesList: FileList = fileInput.files;
            if (filesList.length > 0) {
                let formElement: HTMLFormElement = document.getElementById('upload-form') as HTMLFormElement;
                formElement.submit();
            }
        }
    );
}
