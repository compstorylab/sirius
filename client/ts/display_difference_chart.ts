import * as d3 from "d3";
import * as $ from "jquery";
import {Color} from "./styles";

import {Drawer} from './drawer';
import {createPlot} from "./plots";

let generated_files_path = '/static/generated_files/';
/**
 * when click an edge, higlight the edge and the corresponding pair of nodes. if click on another edge, return the previous
 * clicked one into original color.
 * @param clickedElementList: array, [[source_node_index, target_node_index, edge_dom_obj], [...] ]
 * @param nodes: d3 selection obj, returned from d3.selectAll("circle")
 * @param sourceIndex: int, index of the source node
 * @param targetIndex: int, index of the target node
 * @param clickedElement: obj, the clicked edge, which is a line svg element 
 */
function clickHighlight(clickedElementList, nodes, sourceIndex, targetIndex, clickedElement){
    // get the previously clicked item, if it is not empty, turn it to original color
    let clickedItemsArray = clickedElementList.pop();
    if (clickedItemsArray){
        let previousClickedSourceIndex = clickedItemsArray[0],
            previousClickedTargetIndex = clickedItemsArray[1],
            previousClickedEdge = clickedItemsArray[2];
        nodes.nodes()[previousClickedSourceIndex].style.fill = Color.White;
        nodes.nodes()[previousClickedSourceIndex].setAttribute('stroke', Color.White);
        nodes.nodes()[previousClickedTargetIndex].style.fill = Color.White;
        nodes.nodes()[previousClickedTargetIndex].setAttribute('stroke', Color.White);
        previousClickedEdge.setAttribute('stroke', Color.White);
    }

    // save the clicked items info into an array
    clickedElementList.push([sourceIndex, targetIndex, clickedElement]);
    //highlight the clicked edge, related pairs of nodes
    nodes.nodes()[sourceIndex].style.fill = Color.Blue;
    nodes.nodes()[sourceIndex].setAttribute('stroke', Color.Blue);
    nodes.nodes()[targetIndex].style.fill = Color.Blue;
    nodes.nodes()[targetIndex].setAttribute('stroke', Color.Blue);
    clickedElement.setAttribute('stroke', Color.Blue);
}

/**
 * when user click on an edge, display the static chart corresponding to the pair of nodes if the chart is available. if not
 * display a message.
 * highlight the clicked edge and its nodes.
 */
export function displayChart(){
    // add event listener to the document, ensure the listener can be created before the target element, lke line, is created
    document.addEventListener("click", function(e){
        let clickedElement: any = e.target;

        if (clickedElement.nodeName=='line'){
            let sourceName = clickedElement.__data__.source.name,
                sourceIndex = clickedElement.__data__.source.index,
                targetName = clickedElement.__data__.target.name,
                targetIndex = clickedElement.__data__.target.index,
                plotType = clickedElement.__data__.viztype;


            // Clear previous plots first
            // @ts-ignore
            Plotly.purge('plot-parent');
            clearImage(document.getElementById("difference_chart") as HTMLImageElement);

            // Select Plot by type
            if(plotType == 'DD') {
                loadPlotylPlot(plotType, sourceName, targetName, sourceIndex, targetIndex, clickedElement);
            }
            else if(plotType == 'CD' || plotType == 'DC' || plotType == 'CC') {
                loadPNGGraph(sourceName, targetName, sourceIndex, targetIndex, clickedElement);
            }
        }
    });
}

function setImage(imageElement:HTMLImageElement, staticImageURL:string){
    imageElement.src = staticImageURL;
    imageElement.width = document.querySelector(".right-bar-image").clientWidth;
    imageElement.height = imageElement.width * 2/3;

    console.log("imageElement", imageElement);
    console.log("imageElement size", imageElement.width, imageElement.height);
}

function clearImage(imageElement:HTMLImageElement) {
    imageElement.removeAttribute("src");
    imageElement.removeAttribute("height");
    imageElement.removeAttribute("width");
}


function loadPNGGraph(sourceName:string, targetName:string, sourceIndex:number, targetIndex:number, clickedElement:HTMLElement){
    let clickedElementList = [];
    let imageType = '.png';
    let staticImageFileName = sourceName + "_" + targetName + imageType;
    // let uploadFolderPath = '/static/upload_files/';
    let uploadFolderPath = generated_files_path;

    let staticImageURL = uploadFolderPath + staticImageFileName;
    let imageElement:HTMLImageElement = document.getElementById("difference_chart") as HTMLImageElement;
    let nodes = d3.selectAll("circle");

    // let imageTitle = sourceName.toUpperCase() + ' VS ' + targetName.toUpperCase();
    let imageTitle = sourceName.split("_").join(" ").toUpperCase() + ' VS ' + targetName.split("_").join(" ").toUpperCase();
    Drawer.getInstance().open({mode:'plot', title: imageTitle});

    $.ajax({
        url: staticImageURL,
        success: function(){
            setImage(imageElement, staticImageURL);

            // highlight currently clicked nodes, turn the previous clicked items into original color
            clickHighlight(clickedElementList, nodes, sourceIndex, targetIndex, clickedElement);

        },
        error: function(){
            // highlight currently clicked nodes, turn the previous clicked items into original color
            clickHighlight(clickedElementList, nodes, sourceIndex, targetIndex, clickedElement);

            // todo: in the future, using another way to handle missing pictures?
            let errorMsg:string = "No Image For this Edge";
            Drawer.getInstance().open({mode:'plot', title: errorMsg});

            clearImage(imageElement);
        }
    });
}

/**
 *
 * @param {string} type
 * @param {string} sourceName
 * @param {string} targetName
 * @param {number} sourceIndex
 * @param {number} targetIndex
 * @param {HTMLElement} clickedElement
 */
function loadPlotylPlot(type:string, sourceName:string, targetName:string, sourceIndex:number, targetIndex:number, clickedElement:HTMLElement){
    loadJson(sourceName, targetName,
        function (data){
            let imageTitle = sourceName.split("_").join(" ").toUpperCase() + ' VS ' + targetName.split("_").join(" ").toUpperCase();
            Drawer.getInstance().open({mode:'plot', title: imageTitle});
            createPlot(type, data, 'plot-parent');
        },
        function () {
            Drawer.getInstance().open({mode:'plot', title: "Could not load data to plot."});
        });

    let clickedElementList = [];
    let nodes = d3.selectAll("circle");
    clickHighlight(clickedElementList, nodes, sourceIndex, targetIndex, clickedElement);
}


/**
 * Tries to load the JSON by alternating node names.
 * It is not clear which name should be first from the graph.
 * @param {string} sourceName
 * @param {string} targetName
 * @param onComplete Callback function. function(data:any):void
 * @param onError Callback function function():void
 */
function loadJson(sourceName: string, targetName: string, onComplete, onError) {
    let uploadFolderPath = generated_files_path;
    // let uploadFolderPath = '/static/upload_files/';
    $.ajax({ url: uploadFolderPath + sourceName + "_" + targetName + ".json" })
        .done(function (data) {
            onComplete(data);
        })
        .fail(
        function () {
            return $.ajax({ url: uploadFolderPath + targetName + "_" + sourceName + ".json" })
                .done(function (data) {
                    onComplete(data);
                })
                .fail(function () {
                    onError()
                })
        }
    );

}
