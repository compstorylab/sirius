import { createStore } from 'redux'

const startState:any = {
  graphData: null
}
export let store:any = createStore(siriusReducer, startState);

function siriusReducer(state = 0, action) {
  switch (action.type) {
    case SAVE_GRAPH_DATA:
      return Object.assign({}, state, {graphData: action.data})
    default:
      return state
  }
}

export const SAVE_GRAPH_DATA = "SAVE_GRAPH_DATA"
export function saveGraph(graphData:any) {
  return {
    type: SAVE_GRAPH_DATA,
    data: graphData
  }
}