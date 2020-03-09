import * as d3 from "d3";
import * as $ from "jquery";
import {Color} from "./styles";


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
    let clickedElementList = [];
    // add event listener to the document, ensure the listener can be created before the target element, lke line, is created
    document.addEventListener("click", function(e){
        let clickedElement: any = e.target;
        let nodes = d3.selectAll("circle");

        if (clickedElement.nodeName=='line'){
            let sourceName = clickedElement.__data__.source.name,
                sourceIndex = clickedElement.__data__.source.index,
                targetName = clickedElement.__data__.target.name,
                targetIndex = clickedElement.__data__.target.index,
                imageType = '.png',
                uploadFolderPath = '/static/upload_files/';
            let staticImageFileName = sourceName + "_" + targetName;
            staticImageFileName = staticImageFileName + imageType;

            let staticImageURL = uploadFolderPath + staticImageFileName,
                rightContentBar = <HTMLElement>document.querySelector(".right-bar-content"),
                imageElement = <HTMLImageElement>document.getElementById("difference_chart"),
                rightImageBar = <HTMLElement>document.querySelector(".right-bar-image"),
                uploadLink = document.getElementById("upload-link"),
                imageTitleElement = document.getElementById("image-title"),
                rightBar = document.getElementById("right-bar");

                rightBar.hidden = false; // display right bar
                rightContentBar.hidden = true;
                rightImageBar.hidden = false; // display image bar, but hide conntent bar

            $.ajax({
                url: staticImageURL,
                success: function(){
                    // let imageTitle = sourceName.toUpperCase() + ' VS ' + targetName.toUpperCase();
                    let imageTitle = sourceName.split("_").join(" ").toUpperCase() + ' VS ' + targetName.split("_").join(" ").toUpperCase();
                    imageTitleElement.innerText = imageTitle;
                    imageElement.src = staticImageURL;
                    imageElement.width = document.querySelector(".right-bar-image").clientWidth;
                    imageElement.height = imageElement.width * 2/3;
                    console.log("imageElement", imageElement);
                    console.log("imageElement size", imageElement.width, imageElement.height);

                    // highlight currently clicked nodes, turn the previous clicked items into original color
                    clickHighlight(clickedElementList, nodes, sourceIndex, targetIndex, clickedElement);

                },
                error: function(){
                    // highlight currently clicked nodes, turn the previous clicked items into original color
                    clickHighlight(clickedElementList, nodes, sourceIndex, targetIndex, clickedElement);
                    // todo: in the future, using another way to handle missing pictures?
                    imageTitleElement.innerText = "No Image For this Edge";
                    imageElement.removeAttribute("src");
                    imageElement.removeAttribute("height");
                    imageElement.removeAttribute("width");
                }
            });

        }
    });

}