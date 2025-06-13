
"""
rag_agent.py

This module defines the RAG (Retrieval-Augmented Generation) agent node
for the LangGraph legal research agent.

It retrieves relevant legal documents from a vector DB (FAISS) using
semantic similarity and combines them with the user's query to
generate a context-aware response.

It follows this pattern:
Retrieve context → PromptTemplate → LLM → StrOutputParser

Functions:
    rag_agent(state: AgentState) -> AgentState
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from nodes.supervisor import AgentState
from config import DEFAULT_MODEL, MODEL_TEMPERATURE
from utils.embedding_utils import load_vectorstore


# Step 1: Load retriever
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Step 2: Define prompt template with context placeholder
rag_prompt = PromptTemplate.from_template(
    """
You are a highly knowledgeable and reliable legal assistant.

Provide a clear, concise, and legally accurate response to the user's question based solely on the provided context. 
Do not speculate, add external information, or rely on prior knowledge beyond the context. 
If the answer is not explicitly present in the provided documents, respond with: "Not found in the provided documents." 
Keep the response professional and directly relevant to the question.

Context:
{context}

Question:
{question}

    """
)

# Step 3: Setup LLM + parser
llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
      temperature=MODEL_TEMPERATURE
      )

parser = StrOutputParser()
rag_chain = rag_prompt | llm | parser


# Step 4: Node function
def rag_agent(state: AgentState) -> AgentState:
    """
    Retrieves legal documents and generates a context-aware answer.
    Updates `intermediate_response` in agent state.
    """
    print("📚 RAG Agent processing query:", state.query)

     # Safety check
    if not retriever:
        raise ValueError("Retriever could not be loaded.")
    
    # Retrieve relevant chunks
    docs = retriever.invoke(state.query)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Chain invocation
    response = rag_chain.invoke({"question": state.query, "context": context})
    state.intermediate_response = response.strip()
    return state