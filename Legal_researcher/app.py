import streamlit as st
from main import graph
from nodes.supervisor import AgentState

st.set_page_config(page_title="Legal Research Agent", layout="centered")
st.title("⚖️ Legal Research Chat Assistant")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.chat_input("Ask your legal question...")

# When the user submits a query
if user_input:
    # Show user message
    st.session_state.chat_history.append(("user", user_input))
    
    # Run graph
    result = graph.invoke({"query": user_input})

    # Convert to AgentState if needed
    if not isinstance(result, AgentState):
        result = AgentState(**result)

    # Decide assistant response
    if result.final_response and result.supervisor_decision is None:
        assistant_response = result.final_response  # Early termination or invalid query
    else:
        assistant_response = result.final_response or "No final response generated."

    # Save assistant message
    st.session_state.chat_history.append(("assistant", assistant_response))

# Display full chat history
for sender, msg in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(msg)
