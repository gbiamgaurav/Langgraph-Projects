
# nodes/supervisor_node.py

from typing import Optional
from pydantic import BaseModel
from langchain_core.runnables import RunnableLambda
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from config import DEFAULT_MODEL, MODEL_TEMPERATURE
from langchain_google_genai import ChatGoogleGenerativeAI

"""
Supervisor Node for Legal Research Agent (LangGraph Entry Point)

Responsibilities:
1. Validate the query using an LLM:
   - Is it a legal query?
   - Is it clear and unambiguous?
2. Prevent infinite loops via retry counter
3. Route valid queries to the Router
4. Stop invalid or irrelevant queries early with a user-friendly message
"""

# ------------------------------
# LLM Output Structure
# ------------------------------

class PreValidationOutput(BaseModel):
    is_legal: bool
    is_clear: bool
    reason: str

class ValidationOutput(BaseModel):
    is_valid: bool
    reason: str

# Define LangGraph-compatible state
class AgentState(BaseModel):
    query: str
    supervisor_decision: Optional[str] = None
    query_type: Optional[str] = None  # NEW
    route: Optional[str] = None
    intermediate_response: Optional[str] = None
    final_response: Optional[str] = None
    validation_output: Optional[ValidationOutput] = None
    retry_count: int = 0

    # ------------------------------
# Prompt + Parser Chain (LangChain-style)
# ------------------------------

validation_prompt = PromptTemplate.from_template(

    """
    You are a legal query validator. Evaluate the user's query to determine:

    1. Is it clearly about a legal topic (law, constitution, court case, FIR, etc.)?
    2. Is it clearly worded, unambiguous, and actionable?

    Respond ONLY in JSON format like:
    {{
    "is_legal": true or false,
    "is_clear": true or false,
    "reason": "<Short justification>"
    }}

    Query:
    {query}
    """
                                                 
)

llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    temperature=MODEL_TEMPERATURE
)

parser = PydanticOutputParser(pydantic_object=PreValidationOutput)

# Chain: Prompt → LLM → Parser
validator_chain = validation_prompt | llm | parser

# ------------------------------
# Supervisor Node Logic
# ------------------------------
def supervisor_node(state: AgentState) -> AgentState:
    """
    The supervisor node performs initial validation of the user's query using an LLM.
    
    Responsibilities:
    - Filters non-legal or out-of-scope queries.
    - Flags unclear or ambiguous queries and triggers retry logic.
    - Enforces a retry limit (default: 2) to prevent infinite loops.
    - Routes validated queries forward to the Router node.
    
    Returns:
        AgentState with either:
            - final_response if validation fails,
            - or supervisor_decision="route_to_router" if successful.
    """
    print("👨‍⚖️ Supervisor received query:", state.query)

    # ✅ Retry limit safeguard
    if state.retry_count >= 2:
        state.final_response = (
            "⚠️ Your query could not be validated after multiple attempts. "
            "Please rephrase and try again later."
        )
        state.supervisor_decision = None
        return state

    # ✅ Run LLM validation
    validated: PreValidationOutput = validator_chain.invoke({"query": state.query})
    print("📋 Pre-validation result:", validated)

    # 🚫 Non-legal → terminate
    if not validated.is_legal:
        state.final_response = (
            "⚠️ We can assist with legal queries only. "
            "Please reframe your question to focus on legal topics such as law, court procedures, or the constitution."
        )
        state.supervisor_decision = None
        return state

    # ⚠️ Ambiguous → retry with counter
    if not validated.is_clear:
        state.retry_count += 1
        state.final_response = (
            "⚠️ Your query seems unclear or ambiguous. "
            "Please rephrase it more clearly, focusing on specific legal issues."
        )
        state.supervisor_decision = None
        return state
    
    # Decide next step
    state.supervisor_decision = "route_to_router"
    return state