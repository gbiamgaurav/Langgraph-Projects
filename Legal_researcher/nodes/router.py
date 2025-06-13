
"""
router_node.py

This module defines the Router Node using an LLM-based classification approach
for the LangGraph-based legal research agent. It routes the legal query to
'llm', 'rag', or 'crawler' based on structured output parsed from the LLM.

It follows this chaining pattern:
PromptTemplate → LLM → PydanticOutputParser → query_type → state.route
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from nodes.supervisor import AgentState
from config import DEFAULT_MODEL, MODEL_TEMPERATURE




# Step 1: Define the Pydantic schema for the parser
class QueryTypeOutput(BaseModel):
    query_type: str  # should be one of 'llm', 'rag', 'crawler'


# Step 2: Define the output parser
output_parser = PydanticOutputParser(pydantic_object=QueryTypeOutput)

# Step 3: Prompt Template
prompt = PromptTemplate.from_template(
    """
    You are a highly knowledgeable legal assistant specializing in query classification.
    Evaluate the following user query and classify it into one of the following categories based on its primary intent:

    Query:  
    {query}

    Categories:  
    - "llm": Queries seeking general legal explanations, definitions, or concepts (e.g., "What is defamation in law?").  
    - "rag": Queries requesting retrieval of specific case law, acts, statutes, or legal documents (e.g., "What does Section 230 of the Communications Decency Act say?").  
    - "web": Queries asking for real-time updates, recent rulings, or the latest legal developments (e.g., "What are the latest data privacy rulings in India?").

    Follow these classification Rules:  
    - Focus on the query’s primary intent. If the query has multiple intents (e.g., explanation and recent updates), prioritize the most dominant intent.  
    - If the query explicitly seeks "latest" or "recent" legal information, classify it as "web" unless it specifies a known document or case.  
    - If the query references a specific legal document, case, or statute, classify it as "rag" even if it asks for explanation.  
    - If the query is ambiguous or doesn’t fit any category, classify it as "llm" and assume a general explanation is needed.  
    - Consider the jurisdiction if specified, as it may affect the classification (e.g., a query about "recent US rulings" is "web", but "US Constitution Article 1" is "rag").

    Respond ONLY with valid JSON in the format:
     ```json
    {{ "query_type": "category" }} 

    Ensure the value of "query_type" is one of the specified categories ("llm", "rag", "web"). Do not include any additional text or comments.


    """
)

# Step 4: Define the model (OpenAI for now)
llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    temperature=MODEL_TEMPERATURE
)


# Step 5: Chain everything
router_chain = prompt | llm | output_parser

# Step 6: Node function
def router_node(state: AgentState) -> AgentState:
    """
    Uses an LLM to classify the legal query and route accordingly.
    """
    print("📬 Router received query:", state.query)

    result: QueryTypeOutput = router_chain.invoke({"query": state.query})
    state.query_type = result.query_type
    state.route = result.query_type  # Align route with query_type
    
    print(f"🧭 LLM classified query as: {state.query_type}")
    return state

