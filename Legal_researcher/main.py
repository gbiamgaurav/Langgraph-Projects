"""
main.py

Entry point for the legal_research_agent using LangGraph.
This sets up the state machine and defines the flow:
supervisor_node → router_node → agent → validator → finalizer
"""

from langgraph.graph import StateGraph, END
from nodes.supervisor import supervisor_node, AgentState
from nodes.router import router_node
from agents.llm_agent import llm_agent
from agents.rag_agent import rag_agent
from agents.web_crawler import web_crawler_agent
from nodes.validation_node import validation_node
from nodes.finalizer import finalizer_node  

# Step 1: Initialize the graph builder
builder = StateGraph(AgentState)

# Step 2: Add all nodes
builder.add_node("supervisor", supervisor_node) 
builder.add_node("router", router_node)
builder.add_node("llm_agent", llm_agent)
builder.add_node("rag_agent", rag_agent)
builder.add_node("web_crawler", web_crawler_agent)
builder.add_node("validator", validation_node)
builder.add_node("finalizer", finalizer_node)

# Step 3: Define the flow
builder.set_entry_point("supervisor")

builder.add_conditional_edges("supervisor", lambda x: x.final_response is not None, {
    True: END,
    False: "router"
})

builder.add_conditional_edges("router", lambda x: x.query_type, {
    "llm": "llm_agent",
    "rag": "rag_agent",
    "web": "web_crawler"
})

builder.add_edge("llm_agent", "validator")
builder.add_edge("rag_agent", "validator")
builder.add_edge("web_crawler", "validator")

builder.add_conditional_edges("validator", lambda x: x.validation_output.is_valid, {
    True: "finalizer",
    False: "supervisor"
})

builder.set_finish_point("finalizer")

# Step 4: Compile the graph
graph = builder.compile()

if __name__ == "__main__":
    from pprint import pprint

    output = graph.invoke({"query": "Can a person apply for anticipatory bail in a cybercrime case?"})

    # If it's not AgentState, convert it

    if not isinstance(output, AgentState):
        output = AgentState(**output)

    print("Query Type:", output.query_type)
    print("Final Response:", output.final_response)


