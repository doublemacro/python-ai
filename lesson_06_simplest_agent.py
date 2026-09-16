import os
from dotenv import load_dotenv
from openai import OpenAI


# avem agentul. trebuie sa-i dam instructiuni despre cum sa ruleze o unealta oferita de noi.

def main():
    load_dotenv()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY")
    )

    messages = []
    # system prompt.
    messages.append({
        "role": "system",
        "content": "You are a medical assistent, only answer medical questions, and at the end tell the user that this is not medical advice.."
    })
    messages.append({
        "role": "user",
        "content": input("you> ")
    })

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

    # ii cerem agentului sa ne raspunda la un mesaj.
    response = client.chat.completions.create(
        messages=messages,
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        tools=tools,
        tool_choice="auto"
    ).choices[0].message


    print(response)


if __name__ == "__main__":
    main()
