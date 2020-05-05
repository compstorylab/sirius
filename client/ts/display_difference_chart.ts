import * as d3 from "d3";
import * as $ from "jquery";

import {Drawer} from './drawer';
import {createPlot} from "./plots";


/**
 * when user click on an edge, display the static chart corresponding to the pair of nodes if the chart is available. if not
 * display a message.
 * highlight the clicked edge and its nodes.
 */
export function displayChart(plotType:string, sourceName:string, targetName:string){
    // Clear previous plots first
    // @ts-ignore
    Plotly.purge('plot-parent');
    clearImage(document.getElementById("difference_chart") as HTMLImageElement);

    loadPlotylPlot(plotType, sourceName, targetName);
}

function setImage(imageElement:HTMLImageElement, staticImageURL:string){
    imageElement.src = staticImageURL;
    imageElement.width = document.querySelector(".right-bar-image").clientWidth;
    imageElement.height = imageElement.width * 2/3;
}

function clearImage(imageElement:HTMLImageElement) {
    imageElement.removeAttribute("src");
    imageElement.removeAttribute("height");
    imageElement.removeAttribute("width");
}


function loadPNGGraph(sourceName:string, targetName:string){
    let clickedElementList = [];
    let imageType = '.png';
    let staticImageFileName = sourceName + "_" + targetName + imageType;
    // @ts-ignore chartPngPath in the global scope. Check the html.
    let uploadFolderPath = chartPngPath;

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
        },
        error: function(){
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
 */
function loadPlotylPlot(type:string, sourceName:string, targetName:string){
    loadJson(sourceName, targetName,
        function (data){
            let imageTitle = sourceName.split("_").join(" ").toUpperCase() + ' VS ' + targetName.split("_").join(" ").toUpperCase();
            Drawer.getInstance().open({mode:'plot', title: imageTitle});
            createPlot(type, data, 'plot-parent');
        },
        function () {
            Drawer.getInstance().open({mode:'plot', title: "Could not load data to plot."});
        });
}


/**
 * Tries to load the JSON by alternating node names.
 * It is not clear which name should be first from the graph.
 * @param {string} sourceName
 * @param {string} targetName
 * @param onComplete Callback function. function(data:any):void
 * @param onError Callback function function():void
 */
function loadJson(sourceName:string, targetName:string, onComplete, onError) {
    // @ts-ignore chartJsonPath in the global scope. Check the html.
    let uploadFolderPath = chartJsonPath;
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
