"""LangGraph Agent"""
import os
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
# from langchain_community.tools.tavily_search import TavilySearch
from langchain_tavily.tavily_search import TavilySearch
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from supabase.client import Client, create_client
from langchain.embeddings import OpenAIEmbeddings
from transcriber import AudioTranscriber

load_dotenv()

@tool 
def get_transcript():
    """
    Retrieves an audio transcript from an S3 bucket using AWS Transcribe.

    This function attempts to retrieve an existing transcription job's result or initiates a new
    transcription job if one doesn't exist for the specified file. It uses predefined
    AWS S3 bucket details, file path, job name, and AWS region.

    Returns:
        str: The transcript of the audio file if successful, or an error message if an exception occurs.

    Raises:
        Exception: Catches and returns any exceptions that occur during the transcription
                   process, such as issues with AWS credentials, bucket access, or
                   Transcribe service errors.
    """
    try:
        BUCKET_NAME = 'impetus-hackathon'
        FILE_PATH = 'call.mp3'
        JOB_NAME = 'test-transcription-jobX'
        REGION = 'us-east-1' 

        transcriber = AudioTranscriber(region_name=REGION)
        transcript = transcriber.get_or_create_transcript(
            job_name=JOB_NAME,
            bucket_name=BUCKET_NAME,
            file_name=FILE_PATH.split('/')[-1]
        )
        return transcript
    except Exception as e:
        return f"Error getting transcript: {str(e)}"
    
@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.
    
    Args:
        query: The search query."""

    search_docs = TavilySearch(max_results=3).invoke({"query": query})
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc['url']}"/>\n{doc['content']}\n</Document>'
            for doc in search_docs['results']
        ])
    return {"web_results": formatted_search_docs}

# load the system prompt from the file
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()


sys_msg = SystemMessage(content=system_prompt)

tools = [web_search,get_transcript]

# Build graph function
def build_graph(provider: str = "open-ai"):
    """Build the graph"""
    llm = ChatOpenAI(model="gpt-4")
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)

    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()

# test
if __name__ == "__main__":
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    # Build the graph
    graph = build_graph(provider="open-ai")
    # Run the graph
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()

