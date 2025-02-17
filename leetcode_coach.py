import os
from dotenv import load_dotenv
from typing import TypedDict, Literal
from typing import Annotated, List
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import Command
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain


from termcolor import colored

# Load environment variables
load_dotenv()

# OpenAI API Key
openai_key = os.environ.get('OPENAI_API_KEY')
tavily_key = os.environ.get('TAVILY_API_KEY')


class State(TypedDict):
    messages: Annotated[List, add_messages] = []
    next: str = ""

def resource_finder(state: State) -> Command[Literal["Supervisor"]]:
    """Finds resources based on the user's query."""

    query = state["messages"][-1].content
    context = state["messages"]

    # Define prompt
    prompt = ChatPromptTemplate.from_template(
        f"""You are a computer science resource finder. 
        You will find resources for the user based on their preferences 
        in the field of computer science and related topics. 
        Use the context to help you build the response. You may also answer
        general questions or miscellaneous queries.
        
        Question: {query}\n"""
    )
    
    # Bind the wrapped callable to the LLM.
    tavily_tool = TavilySearchResults(max_results=5, search_depth="advanced", include_answer=True, include_raw_content=True)
    llm = ChatOpenAI(model="gpt-4o").bind_tools([tavily_tool])

    rag_chain = (
        prompt
        | llm
    )

    @chain
    def tool_chain(user_input: str, config: RunnableConfig):
        input_ = {"user_input": user_input, "context": context}
        ai_msg = rag_chain.invoke(input_, config=config)
        tool_msgs = tavily_tool.batch(ai_msg.tool_calls, config=config)
        return rag_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)

    response = tool_chain.invoke(query)


    return Command(
        update={
            "messages": [
                AIMessage(content=response.content, name="resource_finder")
            ]
        },
        goto="Supervisor",
    )



def coding_problem_generator(state: State) -> Command[Literal["Supervisor"]]:
    """Generates a coding problem based on the user's query."""

    query = state["messages"][-1].content

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
    prompt = ChatPromptTemplate.from_template(
        """"You are a Leetcode-style coding problem generator. 
        You will generate a coding problem for the user to solve 
        based on their preferences. Use the context to help you build the problem.
        Do not provide a solution to the problem unless asked for.

        Context: {context}\n 
        Answer:"""
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Build RAG chain using retriever, prompt, and LLM
    rag_chain = (
        {"context": retriever | format_docs}
        | prompt
        | llm
        | StrOutputParser()
    )


    # Invoke RAG chain to generate coding problem
    response = rag_chain.invoke(query)

    return Command(
        update={
            "messages": [
                AIMessage(content=response, name="problem_generator")
            ]
        },
        goto="Supervisor",
    )


class Router(TypedDict):
    next: Literal["Resource-Finder", "Problem-Generator", "FINISH"]


def supervisor_agent(state: State) -> Command[Literal["Resource-Finder", "Problem-Generator", "__end__"]]:
    """Supervisor agent that manages the conversation between workers."""

    question = state["messages"][-1].content

    # Include the system prompt and the current conversation state in the messages
    members = ["Resource-Finder", "Problem-Generator"]
    system_prompt = (
            "You are a supervisor tasked with managing a conversation between the"
            f" following workers:  {members}. Given the following user request {question}," 
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When you determine a task to be finished,"
            " respond with FINISH."
            " Here are the uses of each worker:\n"
            "1. Resource-Finder: Find resources based on the user's query and handles any general or miscellaneous user queries.\n"
            "2. Problem-Generator: Generate a coding problem based on the user's query.\n"
        )

    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    llm = ChatOpenAI(model="gpt-4o")

    # Use the LLM to decide the next step
    response = llm.with_structured_output(Router).invoke(messages)

    # Extract the next node from the response
    next_node = response.get("next", None)

    if not next_node:
        raise ValueError("Supervisor failed to determine the next step.")

    if next_node == "FINISH":
        next_node = END
    # Return a Command with the target node in the goto field.
    return Command(goto=next_node, update={"next": next_node})

def create_graph():
    workflow = StateGraph(State)

    workflow.add_node("Resource-Finder", resource_finder)
    workflow.add_node("Problem-Generator", coding_problem_generator)
    workflow.add_node("Supervisor", supervisor_agent)

    workflow.add_edge(START, "Supervisor")

    graph = workflow.compile()

    return graph


def main():
    # Create the state graph
    graph = create_graph()

    # Print out the LangGraph as ASCII
    graph.get_graph().print_ascii()

    # Continuous input and LLM interaction
    print(colored("You can start interacting with the coding assistant. Type 'exit' to end the conversation.", "blue"))

    while True:
        user_message = input("> ")

        if user_message == "exit":
            print(colored("Goodbye!", "blue"))
            break

        input_state = {"messages": [{"role": "user", "content": user_message}]}
        config = {"configurable": {"thread_id": 42}}

        # # Verbose output
        # for event in graph.stream(input_state, config):
        #     print(colored(event, "red"))
        #     print("------------------------------------")

        # Concise output
        final_state = graph.invoke(input_state, config)
        print(colored(final_state["messages"][-1].content, "red"))
        print("------------------------------------")


if __name__ == "__main__":
    main()
 


