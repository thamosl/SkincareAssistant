
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangGraph and Langchain imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import config

load_dotenv()

# --- Configuration and Initialization ---

# Set Groq API Key only when a real value is present.
groq_api_key = os.getenv("GROQ_API_KEY") or getattr(config, "GROQ_API_KEY", "")
if groq_api_key:
    groq_api_key = groq_api_key.strip().strip('"').strip("'")
    os.environ["GROQ_API_KEY"] = groq_api_key
else:
    st.warning("GROQ_API_KEY is not set. Add it to your environment or .env file before running the app.")

# Initialize embeddings model
@st.cache_resource
def get_embeddings_model():
    """Caches the HuggingFaceEmbeddings model."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

embeddings = get_embeddings_model()

import chromadb

# Initialize Chroma vector store
@st.cache_resource
def get_vector_store(_embed_func):
    """Caches the Chroma vector store."""
    try:
        client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIRECTORY)
        return Chroma(
            client=client,
            collection_name=getattr(config, "COLLECTION_NAME", "indian_railways"),
            embedding_function=_embed_func
        )
    except Exception as e:
        st.error(f"Error loading ChromaDB. Make sure '{config.CHROMA_PERSIST_DIRECTORY}' exists and is populated. Error: {e}")
        st.stop()

vectordb = get_vector_store(embeddings)

# Initialize the ChatGroq model
@st.cache_resource
def get_chat_model():
    """Caches the ChatGroq model."""
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=1,
        max_tokens=None
    )

model = get_chat_model()

# Define the LangGraph node function
def call_model(state: MessagesState):
    """
    This function defines the 'model' node in the LangGraph workflow.
    It takes the current state (conversation messages) and invokes the LLM.
    """
    system_prompt = (
        "You are a helpful assistant for question-answering tasks based on SkinCare PDF documents.\n"
        "Use the retrieved context provided in the user's prompt to answer the question accurately and concisely.\n"
        "If the answer is not present in the retrieved context, state clearly that you don't know based on the provided documents.\n"
        "Keep the answer factual, clear, and well-structured."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": response}

# Build and compile the LangGraph workflow
@st.cache_resource
def get_langgraph_app():
    """Caches and compiles the LangGraph workflow."""
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

app = get_langgraph_app()

# --- Streamlit UI Setup ---

st.set_page_config(page_title="Skincare Assistant", layout="centered")
st.title("💬 Skincare Assistant")

# Initialize session state for messages and thread ID
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_chat_session"

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Logic ---

if prompt := st.chat_input("Ask any question about Skincare..."):
    # Add user message to chat history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # 1. Retrieve context for the current user question
                docs_with_scores = vectordb.similarity_search_with_score(prompt, k=4)

                if not docs_with_scores:
                    current_context = "No relevant documents found."
                    extracted_sources = ["N/A"]
                    extracted_pages = []
                else:
                    context_chunks = []
                    extracted_sources = []
                    extracted_pages = []

                    for doc, score in docs_with_scores:
                        context_chunks.append(doc.page_content)
                        source_path = doc.metadata.get('source', 'Unknown Document')
                        source_name = os.path.basename(source_path)
                        if source_name not in extracted_sources:
                            extracted_sources.append(source_name)

                        page_num = doc.metadata.get('page')
                        if page_num is not None:
                            try:
                                display_page = int(page_num) + 1
                                if display_page not in extracted_pages:
                                    extracted_pages.append(display_page)
                            except (ValueError, TypeError):
                                pass

                    current_context = "\n\n---\n\n".join(context_chunks)

                # 2. Construct the HumanMessage for the current turn, including context
                current_turn_message = HumanMessage(content=f"Context:\n{current_context}\n\nQuestion: {prompt}")

                # 3. Invoke the LangGraph app with the new message and thread_id
                result = app.invoke(
                    {"messages": [current_turn_message]},
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                )

                ai_response = result['messages'][-1].content

                source_document = ", ".join(extracted_sources) if extracted_sources else "N/A"
                page_numbers_str = ", ".join(map(str, sorted(extracted_pages))) if extracted_pages else "N/A"

                final_response = f"{ai_response}\n\n**Source Document**: {source_document}\n**Reference Page Numbers**: {page_numbers_str}"

                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})

            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "I encountered an error. Please try again."})

