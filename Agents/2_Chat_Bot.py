import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv 
load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]] # Union allows you to either store Human or AI message in the messages

llm = ChatOpenAI(model="gpt-4o")


def process(state: AgentState) -> AgentState:
    """This node will solve ther equest you input"""
    response = llm.invoke(state["messages"])
    
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state["messages"])

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()

conversation_history = [] #new

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]

    user_input = input("Enter: ")


with open("Agents/2a_chatlog.txt", "w") as file: #Check this to make sure it saves properly
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to 2a_chatlog.txt")

#The code as is will incurr the use of a lot of tokens a possible solution for next time is to delete the very first message when you reach 5 messages.