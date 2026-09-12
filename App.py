from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent


search = GoogleSerperAPIWrapper()

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

tools = [search.run]

system_prompt = "You are an assistant and can search user queries."

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)


while True:
    query = input("User: ")

    if query == "quit":
        break

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    })

    answer = response["messages"][-1].content

    print("AI:", answer)