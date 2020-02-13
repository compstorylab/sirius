import * as d3 from "d3";
import {Color} from "./styles";

/**
 * generate a network graph representing the connections between the features
 * @param jsonUrl: string, the url to the data file in json format
 */
export function generateGraghChart(jsonUrl){
    // let svg = d3.select('svg'),
    //     width = document.getElementById("drawing-section").clientWidth,
    //     height = document.getElementById("drawing-section").clientHeight,
    //     nodeRadius = 10;
    // svg.attr("width", width).attr("height", height);
    let svg = d3.select('svg'),
        height = +svg.attr("height"),
        width = +svg.attr("width"),
        nodeRadius = 8;

    let simulation = d3.forceSimulation()
        .force('charge', d3.forceManyBody())
        .force('center', d3.forceCenter(width/2, height/2))
        // .force('collide', d3.forceCollide().radius(nodeRadius*1.2))
        .force('collide', d3.forceCollide())
        .force('link', d3.forceLink().distance(80)) // distance sets the length of each link
        .force('link', d3.forceLink().id(function(d){ return d.name; }));  // if want to source and targe value in links not using number based zero but based on cusomized string

    d3.json(jsonUrl.value, function(error, data){
        console.log("data?", data)
        let links = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .enter()
            .append("line")
            .attr("stroke", Color.White)
            .attr("stroke-width", 1);

        let nodes = svg.append("g")
            .selectAll("circle")
            .data(data.nodes)
            .enter()
                .append("circle")
                .attr("r", nodeRadius)
                // .style("opacity", "0.5");
                .style('fill',  Color.White)
            .on("mouseover", nodeMouseOverBehavior)
            .on("mouseout", nodeMouseOutBehavior)
            .on("click", nodeClickBehavior);

        simulation.nodes(data.nodes)
            .on("tick", ticked);
        simulation.force("link").links(data.links);

        function ticked(){
            links
                .attr("x1", function(d:any) { return d.source.x; })
                .attr("y1", function(d:any) { return d.source.y; })
                .attr("x2", function(d:any) { return d.target.x; })
                .attr("y2", function(d:any) { return d.target.y; });
            nodes
                .attr("cx", function(d:any) { return d.x; })
                .attr("cy", function(d:any) { return d.y; })
                .attr("stroke",  Color.White)
                .attr("stroke-width", 1);
        }
    });

}

/**
 * a function for mouse over behavior of the node in network graph. When user hovers over the node, display the feature
 * name
 * @param d: data, from the <circle> element's data attributes
 * @param i: index
 */
function nodeMouseOverBehavior(d, i){
    let svg = d3.select('svg');
    svg.append("text")
        // .attr('id', d.source + '_index_' + d.index)
        .attr('id', 'text_' + d.name)
        .attr('x', function() { return d.x - 50})
        .attr('y', function(){ return d.y - 12})
        // .attr('class', 'node-text')
        // .text(function(){ return d.source; })
        .text(function(){ return d.name; })
        .style('fill', 'white')  // need Design QA: font color, size, family, position?
    ;
}

/**
 * a function for mouse out behavior of the node in network graph. When the user moves the mouse out of node, remove the
 * feature name
 * @param d: data, from the <circle> element's data attributes
 * @param i: index
 */
function nodeMouseOutBehavior(d, i){
    // select by id
    // d3.select("#" + d.source + '_index_' + d.index).remove();
    d3.select("#text_" + d.name).remove();
}

/**
 * if the node is clicked, grey out non-neighbors node.
 * @param d datum from the element, here is <circle>'s data 
 * @param i index
 */
function nodeClickBehavior(d, i){
    // start over with all nodes shown
    d3.selectAll("circle")
        .style("opacity", 1);

    // grey out neighbors
    if (d.neighbors){
        let neighborNode = d.neighbors,
            ownName = d.name;
        // grey out non-neighbors
        d3.selectAll("circle")
            .filter(function(d, i){
                if  (d.name == ownName){
                    return false;
                }
                return !(neighborNode.includes(d.name));
            })
            .style("opacity", 0.2);
    }
}

// todo: when click on white space, all ndoes' opacity to 1