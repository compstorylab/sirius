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
        .force('link', d3.forceLink().distance(80)); // distance sets the length of each link

    d3.json(jsonUrl.value, function(error, data){
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
                .style('fill',  Color.White);

        simulation.nodes(data.nodes)
            .on("tick", ticked);
        simulation.force("link").links(data.links);

        function ticked(){
            links
                .attr("x1", function(d) { return d.source.x; })
                .attr("y1", function(d) { return d.source.y; })
                .attr("x2", function(d) { return d.target.x; })
                .attr("y2", function(d) { return d.target.y; });
            nodes
                .attr("cx", function(d) { return d.x; })
                .attr("cy", function(d) { return d.y; })
                .attr("stroke",  Color.White)
                .attr("stroke-width", 1);


        }
    });

}