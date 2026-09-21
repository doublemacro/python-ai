import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


def run_tests() -> str:
    try:
        result = subprocess.run(
            ["uv", "run", "pytest"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=60
        )
        return (result.stdout or "No output.") + "" + (result.stderr or "")
    except Exception as e:
        print(e)

def complete(client: OpenAI, messages: list[dict], tools: list[dict]):
    response = client.chat.completions.create(
        messages=messages,
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        tools=tools,
        tool_choice="auto"
    ).choices[0].message
    return response

# avem agentul. trebuie sa-i dam instructiuni despre cum sa ruleze o unealta oferita de noi.

def main():
    load_dotenv()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY")
    )

    # aici o sa definim lista de tools pentru agent.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_tests",
                "description": "Run uv run pytest."
            }
        }
    ]

    messages = []
    # system prompt.
    messages.append({
        "role": "system",
        "content": "You are a helpful assistant. If the user asks to run some tests, please only run the run_tests tool that you have at your disposal."
    })
    while True:
        inp = input("you> ")
        if inp == "quit":
            break
        messages.append({
            "role": "user",
            "content": inp
        })

        # ii cerem agentului sa ne raspunda la un mesaj.
        response = complete(client, messages, tools)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                output = run_tests()

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output
                })

            response = complete(client, messages, tools)

        print("agent> ", response)
        print(f"Message count: {len(messages)}")


if __name__ == "__main__":
    main()
    # print(run_tests())
