import operator
import requests
from typing import TypedDict, Annotated, Sequence

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage

from strands import Agent
from strands.models.litellm import LiteLLMModel

# Setup environment vars (you can also do this externally)
import os
os.environ["OPENROUTER_API_KEY"] = "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"
os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"

# Define a reusable strands agent for chat
chat_agent = Agent(
    model=LiteLLMModel(
        model_id="openrouter/openai/gpt-3.5-turbo",
        client_args={
            "api_key": os.environ["OPENROUTER_API_KEY"]
        },
        params={
            "temperature": 0.7,
            "max_tokens": 500
        }
    ),
    system_prompt="You are a helpful assistant that handles chat-oriented queries clearly and concisely."
)

# 1. Define the State for our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_input: str
    next_node: str  # Used for routing/logging

# 2. Define Nodes (with strands)
def strands_chat_node(state: AgentState):
    print("--- ROUTING TO STRANDS CHAT AGENT ---")
    user_input = state["user_input"]
    result = chat_agent(user_input)
    content = result.message.get("content", "[⚠️ No content returned]")
    print(f"Strands Agent Response: {content}")
    return {
        "user_input": "",
        "next_node": "end",
        "messages": [HumanMessage(content=user_input), result.message]
    }

def clarification_node(state: AgentState):
    print("--- ROUTING TO CLARIFICATION ---")
    original_input = state["user_input"]
    clarification_message = f"I can only help with chat requests. Your input was: '{original_input}'."
    print(f"Clarification: {clarification_message}")
    return {
        "user_input": "",
        "next_node": "end",
        "messages": [HumanMessage(content=clarification_message)]
    }

# 3. Define router using external API

def route_question(state: AgentState):
    print("--- DECIDING ROUTE VIA INTENT API ---")
    user_input = state["user_input"]
    intent_api_url = "http://0.0.0.0:8000/detect_intent/"

    try:
        response = requests.post(intent_api_url, json={"query": user_input}, timeout=5)
        response.raise_for_status()
        data = response.json()
        intents = data.get("intents", [])
        print(f"Intent API response for '{user_input}': {data}")

        if "chat" in intents:
            print("Intent 'chat' detected. Routing to chat agent.")
            return "strands_chat_node"
        else:
            print("Intent 'chat' not detected. Routing to clarification.")
            return "clarification_node"

    except Exception as e:
        print(f"[ERROR] Intent detection failed: {e}")
        return "clarification_node"

# 4. Assemble the LangGraph
workflow = StateGraph(AgentState)

workflow.add_node("strands_chat_node", strands_chat_node)
workflow.add_node("clarification_node", clarification_node)

workflow.set_conditional_entry_point(
    route_question,
    {
        "strands_chat_node": "strands_chat_node",
        "clarification_node": "clarification_node",
    }
)

workflow.add_edge("strands_chat_node", END)
workflow.add_edge("clarification_node", END)

app = workflow.compile()

def run_graph(input_text: str):
    initial_state = AgentState(
        messages=[HumanMessage(content=input_text)],
        user_input=input_text,
        next_node=""
    )
    final_state = app.invoke(initial_state)
    print("--- GRAPH EXECUTION COMPLETE ---")

if __name__ == "__main__":
    print("LangGraph Router Agent (using Strands Agents)")
    print("Ensure the intent detection API is running at http://0.0.0.0:8000/detect_intent/")
    print("Type 'exit' to quit.")
    while True:
        cli_input = input("You: ")
        if cli_input.lower() == 'exit':
            break
        if cli_input.strip():
            run_graph(cli_input)
