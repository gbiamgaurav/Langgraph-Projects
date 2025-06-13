
"""
Finalizer Node

This node finalizes the LangGraph workflow by transferring the validated
intermediate response to a final response field. It assumes validation
has passed and no further modification is needed.

Can be extended later for formatting, logging, or response templating.
"""

from langchain_core.runnables import RunnableLambda
def finalize_response(state):
    """
    Copies the validated intermediate_response to final_response.
    """
    state.final_response = state.intermediate_response
    print("✅ Final response set:", state.final_response)
    return state

finalizer_node = RunnableLambda(finalize_response)
