
"""
llm_agent.py

This module defines the LLM agent node for general legal queries in the
LangGraph-based legal research agent.

The LLM agent uses a simple prompt-based generation strategy to provide
explanations or summaries of legal concepts. It is triggered when the router
classifies the query type as 'llm'.

It follows this pattern:
PromptTemplate → LLM → StrOutputParser → intermediate_response

Functions:
    llm_agent(state: AgentState) -> AgentState
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from nodes.supervisor import AgentState
from config import DEFAULT_MODEL, MODEL_TEMPERATURE


# Step 1: Define base prompt for legal explanations
base_prompt = PromptTemplate.from_template(
    """
    You are a highly knowledgeable and reliable legal assistant with expertise in current laws and regulations.

    Provide a clear, concise, and legally accurate response to the following question. Base your answer solely on 
    established legal principles, statutes, case law, or official regulations relevant to the jurisdiction specified (if any). 
    If the question lacks a specified jurisdiction, ask for clarification or assume a general, widely applicable legal framework. 
    Avoid speculation, opinions, or unverified information. 
    If the answer is uncertain or requires specialized legal advice, state that clearly and recommend consulting a licensed attorney. 
    Keep the response brief, professional, and directly relevant to the question.

    Question: {query}

    """
)

# Step 2: Model setup
llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    temperature=MODEL_TEMPERATURE
)

# Step 3: Output parser
parser = StrOutputParser()

# Step 4: Chain
llm_chain = base_prompt | llm | parser


# Step 5: Node function
def llm_agent(state: AgentState) -> AgentState:
    """
    Generates a response for general legal queries using LLM.
    Stores the result in `state.intermediate_response`.
    """
    print("💬 LLM Agent is processing the query...")
    response = llm_chain.invoke({"query": state.query})
    state.intermediate_response = response.strip()
    return state