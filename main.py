# requirements:
# pip install openai strands-sdk

from openai import OpenAI
from strands import Agent
import pprint

# --- CONFIGURATION ---
API_KEY = "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"  # Replace this with your real OpenRouter key
MODEL_NAME = "openai/gpt-3.5-turbo"
BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = "You are an expert in AI agent orchestration. Answer all user queries clearly and concisely."
QUERY = "Explain agent orchestration in less than 200 words."

# --- OpenAI Client for OpenRouter ---
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# --- Strands-Compatible Model Wrapper ---
class OpenRouterModel:
    def __init__(self, model_name: str, default_system_prompt: str = None):
        self.model_name = model_name
        self.default_system_prompt = default_system_prompt

    def converse(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        print("\n🧪 Input Debug (OpenRouterModel.converse):")
        print(f"👉 model_name: {self.model_name}")
        print(f"👉 default_system_prompt: {repr(self.default_system_prompt)}")
        print(f"👉 system_prompt (arg): {repr(system_prompt)}")
        print("👉 messages (arg):")
        pprint.pprint(messages)
        if tool_specs:
            print("👉 tool_specs (arg):")
            pprint.pprint(tool_specs)
        if kwargs:
            print("👉 kwargs (arg):")
            pprint.pprint(kwargs)

        # Flatten any rich content (e.g. [{"text": "..."}])
        def flatten(msg):
            content = msg.get("content")
            if isinstance(content, list):
                # Ensure all parts are strings before joining
                flat_content_parts = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        flat_content_parts.append(str(block["text"]))
                    elif isinstance(block, str):
                        flat_content_parts.append(block)
                    # else: # Optionally log or handle other block types
                    # print(f"⚠️ Unknown block type in message content: {type(block)}")
                msg["content"] = "".join(flat_content_parts)
            elif not isinstance(content, str): # If content is not list and not str, convert to str
                msg["content"] = str(content if content is not None else "")
            return msg

        full_messages = []
        current_system_prompt = system_prompt if system_prompt is not None else self.default_system_prompt
        if current_system_prompt:
            full_messages.append({"role": "system", "content": str(current_system_prompt)}) # Ensure content is string

        processed_messages = []
        for m_orig in messages:
            m = m_orig.copy() # Use .copy() to avoid modifying original messages list
            # Ensure role is present and is a string
            role = str(m.get("role", "user")) # Default to user if role is missing
            
            # Flatten and ensure content is a string
            m = flatten(m) # flatten modifies m in-place for 'content'
            content = str(m.get("content", "")) # Ensure content is string after flattening

            processed_messages.append({"role": role, "content": content})

        full_messages.extend(processed_messages)

        print("\n📬 Messages sent to OpenAI client:")
        pprint.pprint(full_messages)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=full_messages,
                temperature=0.7, # Consider making configurable via self.config or kwargs
                max_tokens=500,  # Consider making configurable
                **kwargs
            )

            print("\n🔍 Raw Model Response (OpenAI client):")
            pprint.pprint(response)

            choice = response.choices[0]
            final_content = None
            if choice.message:
                final_content = choice.message.content
            
            if final_content is None:
                print(f"⚠️ Model response content is None. Finish reason: {choice.finish_reason}")
                final_content_stripped = ""
            else:
                final_content_stripped = str(final_content).strip()

            print(f"\n📤 Content returned by converse: {repr(final_content_stripped)}")
            return final_content_stripped
        except Exception as e:
            print(f"\n❌ Error during OpenAI API call in OpenRouterModel.converse: {type(e).__name__} - {e}")
            # Re-raise the exception so it's caught by the main try-except block
            # and prints the error type and message as per existing logic.
            raise


# --- Helper: Extract text from rich message format ---
def extract_final_text(message):
    print(f"\n🔍 Debug: Input to extract_final_text (type: {type(message)}):")
    pprint.pprint(message)

    if not hasattr(message, 'get'): # Check if it's dictionary-like
        print("⚠️ extract_final_text: message is not dictionary-like, attempting to access .content directly if available")
        if hasattr(message, 'content'):
            content = message.content
        else:
            print("⚠️ extract_final_text: message has no .content attribute either.")
            return "[⚠️ Unexpected message object format]"
    else:
        content = message.get("content")

    print(f"🔍 Debug: Content from message (type: {type(content)}):")
    pprint.pprint(content)

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Ensure all parts are strings before joining
        final_text_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                final_text_parts.append(str(block["text"]))
            elif isinstance(block, str): # If a block is already a string
                final_text_parts.append(block)
            # else: # Optionally log or handle other block types
            #     print(f"⚠️ Unknown block type in content list: {type(block)}")
        return "".join(final_text_parts).strip()
    elif content is None:
        print("⚠️ extract_final_text: content is None.")
        return "[⚠️ Content was None]"
    else: # If content is neither str nor list, try to convert to string
        print(f"⚠️ extract_final_text: Unexpected content type ({type(content)}), attempting str conversion.")
        return str(content).strip()

# --- Main Execution ---
if __name__ == "__main__":
    print("🎬 Running Strands Agent with OpenRouter...")
    print("📨 Query:", QUERY)

    model = OpenRouterModel(MODEL_NAME, default_system_prompt=SYSTEM_PROMPT)
    agent = Agent(model=model)

    try:
        response = agent(
            QUERY,
            system_prompt=SYSTEM_PROMPT
        )

        print("\n✅ Final Agent Response:\n")
        print(extract_final_text(response.message))

    except Exception as e:
        print("\n❌ Error during agent execution:")
        print(type(e).__name__, "-", e)