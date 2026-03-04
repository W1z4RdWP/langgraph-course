import random
from typing import TypedDict, Literal, NotRequired

from langgraph.constants import START, END
from langgraph.graph import StateGraph


# Graph State
class State(TypedDict):
    graph_state: NotRequired[str]  # Позволяет отсутствовать

def node_1(state: State) -> State:
    print('__node_1__')
    current = state.get('graph_state', '')  # Безопасный доступ
    return {'graph_state': current + 'I am'}

def node_2(state: State) -> State:
    print('__node_2__')
    current = state.get('graph_state', '')
    return {'graph_state': current + ' happy!'}

def node_3(state: State) -> State:
    print('__node_3__')
    current = state.get('graph_state', '')
    return {'graph_state': current + ' sad!'}

def decide_mood(state: State) -> Literal['node_2', 'node_3']:
    print('__decide_mood__')
    user_input = state.get('graph_state', '')  # Безопасный доступ
    if random.random() < 0.5:
        return 'node_2'
    return 'node_3'



# Build Graph
builder = StateGraph(State)

builder.add_node('node_1', node_1)
builder.add_node('node_2', node_2)
builder.add_node('node_3', node_3)

builder.add_edge(START, 'node_1')
builder.add_conditional_edges('node_1', decide_mood)
builder.add_edge('node_2', END)
builder.add_edge('node_3', END)

graph = builder.compile()

if __name__ == '__main__':
    result = graph.invoke({'graph_state': 'Hello! My name is Ruslan. '})

    print(result['graph_state'])
