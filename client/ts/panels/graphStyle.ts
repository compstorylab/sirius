export const graphStyle = [
{
  "selector": "core",
  "style": {
    "selection-box-color": "#AAD8FF",
    "selection-box-border-color": "#8BB0D0",
    "selection-box-opacity": "0.5"
  }
}, {
  "selector": "node",
  "style": {
    "width": "10px",
    "height": "10px",
    "label": "data(id)",
    "font-size": "3px",
    "text-valign": "top",
    "text-halign": "center",
    "background-color": "#6F7591",
    "color": "white",
    "overlay-padding": "6px",
    "z-index": "10"
  }
}, {
  "selector": "node.selected",
  "style": {
    "border-width": "2px",
    "border-color": "#AAD8FF",
    "border-opacity": "1.0",
    "opacity": "1.0",
    "background-color": "white",
  }
}, {
  "selector": "edge",
  "style": {
    "opacity": "0.4",
    "line-color": "white",
    "width": "2px",
    "overlay-padding": "3px"
  }
}, {
  "selector": ".highlighted",
  "style": {
    "z-index": "999999"
  }
}, {
  "selector": "node.highlighted",
  "style": {
    "border-width": "2px",
    "border-color": "#566573",
    "border-opacity": "1.0",
    "opacity": "1.0",
    "background-color": "white",
  },
}, {
    "selector": "edge.highlighted",
    "style": {
      "opacity": "0.8",
    }
  }
]