import { createStore } from 'redux'

const startState:any = {
  graphData: null,
  filterOptions: {
    nodeIds: [],
    nodeTypes: [],
    edgeWeightExtents: [0, 20]
  }, // Options for rendering filter options
  filterSelections: null, // Selections fro the filter form
  chartInfo: null, // Info needed to load and display the chart
}

export let store:any = createStore(siriusReducer, startState);

function siriusReducer(state = 0, action) {
  switch (action.type) {
    case SAVE_GRAPH_DATA:
      return Object.assign({}, state, {graphData: action.data})
    case SAVE_FILTER_OPTIONS:
      return Object.assign({}, state, {filterOptions: action.data})
    case SAVE_FILTER_SELECTIONS:
      return Object.assign({}, state, {filterSelections: action.data})
    case SAVE_CHART_INFO:
      return Object.assign({}, state, {chartInfo: action.data})
    default:
      return state
  }
}

export const SAVE_GRAPH_DATA = "SAVE_GRAPH_DATA"
export const SAVE_FILTER_OPTIONS = "SAVE_FILTER_OPTIONS"
export const SAVE_FILTER_SELECTIONS = "SAVE_FILTER_SELECTIONS"
export const SAVE_CHART_INFO = "SAVE_CHART_INFO"

export function saveGraph(graphData:any) {
  return {
    type: SAVE_GRAPH_DATA,
    data: graphData
  }
}

export function saveFilterOptions(filterOptions:any) {
  return {
    type: SAVE_FILTER_OPTIONS,
    data: filterOptions
  }
}

export function saveFilterSelections(filterSelections:any) {
  return {
    type: SAVE_FILTER_SELECTIONS,
    data: filterSelections
  }
}

export function saveChartInfo(chartInfo:any) {
  return {
    type: SAVE_CHART_INFO,
    data: chartInfo
  }
}

