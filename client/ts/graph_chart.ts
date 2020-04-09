import * as d3 from "d3";
import {Color} from "./styles";

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
    let svg = d3.select('svg'),
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

    // d3.json(jsonUrl.value, function(error, data){
    d3.json(jsonUrl, function(error, data){
        console.log("data",data);
        let links = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .enter()
            .append("line")
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

// todo: when click on reset button, all ndoes' opacity to 1