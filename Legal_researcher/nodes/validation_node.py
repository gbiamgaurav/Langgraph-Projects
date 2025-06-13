
"""
Legal Response Validation Pipeline using LangChain

This script sets up a LangChain-compatible validation pipeline for legal-related LLM responses.
It evaluates the quality and correctness of an AI-generated legal explanation based on
multiple legal criteria, and returns a structured validation output.

Components:
-----------
1. **PromptTemplate (`validation_prompt`)**:
   A carefully constructed prompt asking a legal validation expert (simulated by the LLM) to
   evaluate a response based on:
     - Factual Accuracy
     - Relevance to the Query
     - Completeness
     - Jurisdiction Appropriateness
     - Source Citation

2. **LLM Setup (`ChatOpenAI`)**:
   Uses OpenAI's chat-based LLM defined via configuration settings (`DEFAULT_MODEL`, `MODEL_TEMPERATURE`).

3. **Output Parsing (`PydanticOutputParser`)**:
   Ensures the LLM's response is parsed into a strongly typed, structured format using Pydantic.
   The expected output includes:
     - `is_valid` (bool): Whether the response is valid.
     - `reason` (str): Explanation if the response is invalid.

4. **Prompt Chain (`llm_chain`)**:
   Combines the prompt, model, and parser into a chain:
       validation_prompt | llm | parser

5. **Stateful Function (`validate_response_fn`)**:
   A function that receives a `state` dictionary containing:
     - `query`
     - `intermediate_response`
   It uses the LLM chain to generate validation and returns the state with an additional key:
     - `validation_output`

6. **RunnableLambda (`validation_node`)**:
   Wraps the validation function in a LangChain `RunnableLambda` so it can be used
   as a node in a larger LangChain pipeline or workflow.

Usage:
------
This script is intended to be used in a LangChain pipeline where legal query responses
must be validated for quality assurance or audit purposes.

Example input to `validate_response_fn`:
```python
{
    "query": "What is the statute of limitations for breach of contract in California?",
    "intermediate_response": "In California, the statute of limitations for breach of contract is four years."
}
"""

from langchain_core.runnables import RunnableLambda
from config import DEFAULT_MODEL, MODEL_TEMPERATURE
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from nodes.supervisor import AgentState,ValidationOutput

# Step 1: Define validation prompt for legal explanations
validation_prompt = PromptTemplate.from_template(
    
    """
    You are a legal expert tasked with evaluating the quality of AI-generated legal responses. Your evaluation should be thorough and based on the following criteria:
    Evaluate the following response to the given query based on the criteria below:

    Query:  
    {query}

    Response:  
    {intermediate_response}

    Evaluation Criteria:  
    1. Factual Accuracy: Are all claims in the response factually correct and supported by evidence or reliable knowledge?  
    2. Relevance to Query: Does the response directly address the query and stay on topic?  
    3. Completeness: Does the response fully answer the query, covering all necessary aspects without omitting key details?  
    4. Jurisdiction Appropriateness: Is the response appropriate for the relevant legal, cultural, or regional context of the query?  
    5. Source Citation: If sources are mentioned, are they credible, relevant, and properly cited with clear attribution?

    Response Format:  
    Return your evaluation in the following JSON structure:  
    ```json
    {{
        "is_valid": <true/false>,
        "reason": "<If 'is_valid' is false, provide a clear, concise explanation of the issues based on the criteria. If 'is_valid' is true, leave this field empty or state 'No issues identified.'>"
    }}

    """

)

# Step 2: Model setup
llm = ChatGoogleGenerativeAI(
    model=DEFAULT_MODEL,
    temperature=MODEL_TEMPERATURE
)

parser = PydanticOutputParser(pydantic_object=ValidationOutput)

# Step 4: Build prompt → LLM → parser chain
llm_chain = validation_prompt | llm | parser

# Step 5: Wrap in a function that keeps LangChain state
def validate_response_fn(state: AgentState) -> AgentState:
    result = llm_chain.invoke({
        "query": state.query,
        "intermediate_response": state.intermediate_response
    })
    state.validation_output = result
    state.final_response = result.reason
    print("🔍 Validation Output from validation_node:", result)
    return state

# Step 6: Wrap in a RunnableLambda to be used in a pipeline
validation_node = RunnableLambda(validate_response_fn)