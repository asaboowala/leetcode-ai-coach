import os
from dotenv import load_dotenv
import functools
import operator
from typing import Sequence, TypedDict, Annotated, Literal
from pydantic import BaseModel

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph, START
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.types import Command

from colorama import init
from termcolor import colored

init()
 

load_dotenv()

# OpenAI API Key
openai_key = os.environ.get('OPENAI_API_KEY')
langchain_key = os.environ.get('LANGCHAIN_API_KEY')
tavily_key = os.environ.get('TAVILY_API_KEY')

# Define required tools
tavily_tool = TavilySearchResults(max_results=5)
python_repl_tool = PythonREPLTool()

# Define LLM model
llm = ChatOpenAI(model="gpt-4o")

members = ["Resource-Finder", "Problem-Generator", "Grader"]
options = members + ["FINISH"]
system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " following workers:  {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

@tool
def coding_problem_generator(query: str) -> str:
    """Generates a coding problem based on the user's query."""

    # Using Leetcode dataset for RAG to generate coding problems
    coding_dataset = "leetcode_dataset - lc.csv"

    # Define LLM model
    llm = ChatOpenAI(model="gpt-4o")

    # Load CSV file as input document
    loader = CSVLoader(file_path=coding_dataset)
    docs = loader.load()

    # Split documents into smaller chunks and store in Chroma vectorstore
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Define retriever using vector store
    retriever = vectorstore.as_retriever()

    # Define prompt
    prompt = PromptTemplate(
                input_variables=['context', 'question'], 

                template="""You are a Leetcode-style coding problem generator. 
                You will generate a coding problem for the user to solve 
                based on their preferences. Use the context to help you build the problem.
                Do not provide a solution to the problem unless asked for.

                Question: {question}\n 
                Context: {context}\n 
                Answer:"""
                 
            )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build RAG chain using retriever, prompt, and LLM
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoke RAG chain to generate coding problem
    response = rag_chain.invoke(query)

    return response

def agent_node(state: MessagesState, agent, name) -> Command[Literal["supervisor"]]:
    result = agent.invoke(state)

    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name=name)
            ]
        },
        goto="supervisor",
    )


    # return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}


class Router(TypedDict):
    next: Literal[*options]


# # The agent state is the input to each node in the graph
# class AgentState(TypedDict):
#     # The annotation tells the graph that new messages will always be added to the current states
#     messages: Annotated[Sequence[BaseMessage], operator.add]
#     # The 'next' field indicates where to route to next
#     next: str


def supervisor_agent(state: MessagesState) -> Command[Literal[*members, "__end__"]]:

    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        print("Conversation has ended!!!!!")
        goto = END

    return Command(goto=goto)

    
    

    # # Our team supervisor is an LLM node. It just picks the next agent to process
    # # and decides when the work is completed
    

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", system_prompt),
    #         MessagesPlaceholder(variable_name="messages"),
    #         (
    #             "system",
    #             "Given the conversation above, who should act next?"
    #             " Or should we FINISH? Select one of: {options}",
    #         ),
    #     ]
    # ).partial(options=str(options), members=", ".join(members))

    # supervisor_chain = (
    #     prompt
    #     | llm.with_structured_output(routeResponse)
    # )
    # return supervisor_chain.invoke(state)

def create_graph():
    #Define resource finder agent
    resource_finder_prompt = "You are a helpful resource finder that will search for topics in programming and computer science."
    resource_finder_agent = create_react_agent(llm, tools=[tavily_tool], state_modifier=resource_finder_prompt)
    resource_finder_node = functools.partial(agent_node, agent=resource_finder_agent, name="Resource-Finder")

    #Define problem generator agent
    problem_prompt = "You are a coding problem generator. Use the tool to generate a coding problem based on the user's request and provide the exact answer. Do not provide any solution unless requested by the user."
    problem_agent = create_react_agent(llm, tools=[coding_problem_generator], state_modifier=problem_prompt)
    problem_node = functools.partial(agent_node, agent=problem_agent, name="Problem-Generator")

    #Define grader agent
    grader_prompt = "You are a coding grader. You will run the user's code and provide feedback on their solution based on the coding problem they were provided with such as bugs in the code, syntax/logical errors, etc. Do not overtly give the solution but guide them in the right direction unless they explicitly request a solution."
    grader_agent = create_react_agent(llm, tools=[python_repl_tool], state_modifier=grader_prompt)
    grader_node = functools.partial(agent_node, agent=grader_agent, name="Grader")

    # Define the workflow
    workflow = StateGraph(MessagesState)
    workflow.add_node("Resource-Finder", resource_finder_node)
    workflow.add_node("Problem-Generator", problem_node)
    workflow.add_node("Grader", grader_node)
    workflow.add_node("supervisor", supervisor_agent)

    members = ["Resource-Finder", "Problem-Generator", "Grader"]

    # Add edges to the graph, starting with the supervisor
    workflow.add_edge(START, "supervisor")

    # Add edges to the graph, every worker reports back to the supervisor
    for member in members:
        # We want our workers to ALWAYS "report back" to the supervisor when done
        workflow.add_edge(member, "supervisor")

    # The supervisor populates the "next" field in the graph state which routes to a node or finishes
    # conditional_map = {k: k for k in members}
    # conditional_map["FINISH"] = END
    # workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    # Finally, add entrypoint
    

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph


def main():
    graph = create_graph()

    # Print out the LangGraph as ASCII
    # graph.get_graph().print_ascii()

    # Continuous input and LLM interaction
    print(colored("You can start interacting with the coding assistant. Type 'exit' to end the conversation.", "blue"))

    while True:
        user_message = input("> ")

        if user_message == "exit":
            break

        # input_message = HumanMessage(content=user)
        config = {"configurable": {"thread_id": "1"}}
        # llm_output = graph.invoke(user_message, config)
       

        for event in graph.stream({"messages": [user_message]}, config, subgraphs=True):
            print(colored(event, "red"))
            print("------------------------------------")



    # config = {"configurable": {"thread_id": "1"}}
    # input_message = HumanMessage(content="I want an easy coding problem about arrays. Please give me a solution too with code.")
    # for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    #     if "__end__" not in event:
    #         event["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()


