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
        width = window.innerWidth-50,
        height = window.innerHeight-100,
        nodeRadius = 5;

    let simulation = d3.forceSimulation()
        .force('charge', d3.forceManyBody().distanceMax(height/2))
        .force('center', d3.forceCenter(width/2, height/2))
        // .force('collide', d3.forceCollide().radius(nodeRadius*1.2))
        .force('collide', d3.forceCollide())
        .force('link', d3.forceLink().distance(80)); // distance sets the length of each link

    d3.json(jsonUrl.value, function(error, data){
        let links = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .enter()
            .append("line")
            .attr("stroke", Color.White)
            .attr("stroke-width", 2);

        let nodes = svg.append("g")
            .selectAll("circle")
            .data(data.nodes)
            .enter()
                .append("circle")
                .attr("r", nodeRadius)
                // .style("opacity", "0.5");
                .style('fill',  Color.White)
            .on("mouseover", nodeMouseOverBehavior)
            .on("mouseout", nodeMouseOutBehavior);

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
        .attr('id', d.source + '_index_' + d.index)
        .attr('x', function() { return d.x - 50})
        .attr('y', function(){ return d.y - 12})
        // .attr('class', 'node-text')
        .text(function(){ return d.source; })
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
    d3.select("#" + d.source + '_index_' + d.index).remove();
}