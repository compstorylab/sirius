import * as d3 from "d3";
import {Color} from "./styles";

import {displayChart} from "./display_difference_chart";


/**
 * generate a network graph representing the connections between the features
 * @param jsonUrl: string, the url to the data file in json format
 */
export function generateGraphChart(jsonUrl){
    // let svg = d3.select('svg'),
    //     width = document.getElementById("drawing-section").clientWidth,
    //     height = document.getElementById("drawing-section").clientHeight,
    //     nodeRadius = 10;
    // svg.attr("width", width).attr("height", height);
    let svg = d3.select('#svg-graph'),
        width = window.innerWidth-50,
        height = window.innerHeight-100,
        nodeRadius = 5;

    let simulation = d3.forceSimulation()
        .force('charge', d3.forceManyBody().distanceMax(height/2))
        .force('center', d3.forceCenter(width/2, height/2))
        // .force('collide', d3.forceCollide().radius(nodeRadius*1.2))
        .force('collide', d3.forceCollide())
        .force('link', d3.forceLink().distance(80)) // distance sets the length of each link
        .force('link', d3.forceLink().id(function(d){ return d.name; }));  // for source and targe value in links not using number based zero but based on customized string

    d3.json(jsonUrl.value, function(error, data){
        // These thicker lines make it easier for someone to click on a graph edge.
        let bg_links = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .enter()
            .append("line")
            .attr("stroke", Color.White)
            .attr("stroke-width", 4)
            .attr('stroke-opacity', 0.0)
            .on('click', onLinkClick);


        let links = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .enter()
            .append("line")
            .attr("pointer-events", "none")
            .attr("stroke", Color.White)
            .attr("stroke-width", 2)
            .attr('stroke-opacity', 0.75);


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
        simulation.force("bg_link").links(data.links);

        function ticked(){
            bg_links
                .attr("x1", function(d:any) { return d.source.x; })
                .attr("y1", function(d:any) { return d.source.y; })
                .attr("x2", function(d:any) { return d.target.x; })
                .attr("y2", function(d:any) { return d.target.y; });
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
    let svg = d3.select('#svg-graph');

    let avgCharWidth = 7;
    let textWidth = d.name.length * avgCharWidth;
    let boxWidth = textWidth + 20;

    let popOverGroup = svg.append('g')
        .attr('id', 'text_' + d.name)
        .attr('transform', function() {
            return `translate(${d.x},${d.y-30})`});

    popOverGroup.append('rect')
        .attr('width', boxWidth)
        .attr('height', 20)
        .attr('x', -0.5 * boxWidth)
        .attr('rx', 5)
        .attr('fill', "black")
        .style('fill-opacity', 0.9);

    popOverGroup.append("text")
        .text(function(){ return d.name; })
        .attr('x', 0)
        .attr('y', 15)
        .style('text-anchor', 'middle')
        .style('fill', 'white')
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
    d3.select("#text_" + d.name).remove();
}

/**
 * if the node is clicked, grey out non-neighbors node.
 * @param d datum from the element, here is <circle>'s data 
 * @param i index
 */
function nodeClickBehavior(d, i){
    let svg = d3.select('#svg-graph');
    let circles = svg.selectAll("circle");

    // Reset styles
    svg.selectAll('line')
        .style('opacity', 1.0)
        .style('stroke', Color.White);
    circles.style("opacity", 1)
           .style('stroke', Color.White)
           .style('fill', Color.White);

    // Grey out neighbors
    if (d.neighbors){
        let neighborNode = d.neighbors,
            ownName = d.name;
        // grey out non-neighbors
        circles.filter(function(d, i){
                if  (d.name == ownName){
                    return false;
                }
                return !(neighborNode.includes(d.name));
            })
            .style("opacity", 0.2);
    }

}

/**
 * Handle edge clicks
 * @param d datum from the element
 * @param i index
 */
function onLinkClick(d, i) {
    let source = d.source;
    let target = d.target;
    let vizType = d.viztype;


    let svg = d3.select('#svg-graph');
    let lines = svg.selectAll('line');
    let circles = svg.selectAll("circle");

    // Reset styles
    svg.selectAll('line')
        .style('opacity', 1.0)
        .style('stroke', Color.White);
    circles.style("opacity", 1)
           .style('stroke', Color.White)
           .style('fill', Color.White);


    // Color the nodes defining the edge blue
    circles.filter(function (d) {
            return d.name === source.name || d.name === target.name;
        })
        .style('opacity', 1.0)
        .style('stroke', Color.Blue)
        .style('fill', Color.Blue);
    // Color the edge blue
    lines.filter(function (d) {
            return d.source.name === source.name && d.target.name === target.name
        })
        .style('opacity', 1.0)
        .style('stroke', Color.Blue);

    // Load and display the chart
    displayChart(vizType, source.name, target.name);

}

// todo: when click on reset button, all ndoes' opacity to 1