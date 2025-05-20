# requirements:
# pip install strands-agents[litellm] strands-agents-tools

import os
from strands import Agent
from strands.models.litellm import LiteLLMModel
from strands_tools import calculator

# --- Set environment variables for OpenRouter ---
os.environ["OPENROUTER_API_KEY"] = "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"  # Replace with your real OpenRouter API key
os.environ["OPENROUTER_API_BASE"] = "https://openrouter.ai/api/v1"

# --- Configuration ---
MODEL_ID = "openrouter/openai/gpt-3.5-turbo"  # Use OpenRouter-prefixed model ID
SYSTEM_PROMPT = (
    "You are a tool-only agent. Only respond if a tool successfully completes the user's request. "
    "If no tool handles the request, respond with: 'I'm unable to help with that using the available tools.'"
)
QUERY = "what is 5 plus 7"

# --- Initialize LiteLLMModel ---
model = LiteLLMModel(
    client_args={
        "api_key": os.environ["OPENROUTER_API_KEY"]
    },
    model_id=MODEL_ID,
    params={
        "max_tokens": 500,
        "temperature": 0.7,
    }
)

# --- Create the agent ---
agent = Agent(model=model, tools=[calculator])

# --- Run the agent ---
print("\n🎬 Running Agent with LiteLLMModel via OpenRouter...")
print("📨 Query:", QUERY)

try:
    response = agent(
        QUERY,
        system_prompt=SYSTEM_PROMPT
    )

    tools_invoked = getattr(response, "tools_used", [])
    if not tools_invoked:
        print("\n🚫 No tools were used. Rejecting fallback response.")
        print("🤖 I'm unable to help with that using the available tools.")
    else:
        print("\n✅ Final Agent Response:\n")
        print(response.message.get("content", "[⚠️ No content returned]"))

except Exception as e:
    print("\n❌ Error during agent execution:")
    print(type(e).__name__, "-", e)