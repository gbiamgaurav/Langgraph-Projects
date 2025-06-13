
"""
web_crawler.py

Web crawling agent that fetches real-time legal updates,
summarizes them, and generates a response using an LLM.

Pattern: Web content → PromptTemplate → LLM → OutputParser

Functions:
    web_crawler_agent(state: AgentState) -> AgentState
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from nodes.supervisor import AgentState
from config import DEFAULT_MODEL, MODEL_TEMPERATURE
from utils.web_utils import fetch_legal_webpage

# Example legal news page (can be made dynamic via query routing logic)
DEFAULT_URL = "https://www.livelaw.in/top-stories"

# Prompt Template
web_prompt = PromptTemplate.from_template(
    """
    
    You are a highly knowledgeable and reliable legal analyst AI.

    Summarize the most recent legal developments relevant to the user’s query, based solely on the provided web content. 
    Ensure the developments are current , by checking for explicit dates in the content; if no dates are available or the content is outdated, 
    state: "No recent legal developments found in the provided content as of current month" Only include information explicitly stated in the web content—
    do not speculate, infer, or add external knowledge. If the content is not relevant to the query or lacks legal developments, 
    state: "The provided content does not address the query or contain relevant legal developments." 
    Verify the credibility of the source (e.g., government sites, reputable legal publications) and note if the source appears unreliable. 
    If the query specifies a jurisdiction, focus on developments in that jurisdiction; otherwise, ask the user to clarify the jurisdiction. 
    Provide a clear, concise, and professional response tailored for a legal audience, including citations or references to the source where applicable.

    Web Content:
    {content}

    User Query:
    {query}
    
    """

)

# LLM & Chain
llm = ChatGoogleGenerativeAI(
     model=DEFAULT_MODEL,
       temperature=MODEL_TEMPERATURE
       )

parser = StrOutputParser()
web_chain = web_prompt | llm | parser


# Node function
def web_crawler_agent(state: AgentState) -> AgentState:
    """
    Fetches web content and uses LLM to generate an informed response.
    Updates `intermediate_response` in agent state.
    """
    print("🌐 Web Crawler Agent fetching info for query:", state.query)

    web_content = fetch_legal_webpage(DEFAULT_URL)
    response = web_chain.invoke({"query": state.query, "content": web_content})
    state.intermediate_response = response.strip()
    return state