import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate, Settings, set_global_tokenizer
from transformers import AutoTokenizer
from datetime import datetime
from llama_index.core.memory import ChatMemoryBuffer
import time

# Define the data directory for loading documents
DATA_DIR = "docs"

# Mengecek apakah 'is_initialized' sudah ada di session state
if 'is_initialized' not in st.session_state:
    st.session_state.is_initialized = False

# Inisialisasi yang hanya dilakukan sekali saat pertama kali load
if not st.session_state.is_initialized:
    st.session_state.is_initialized = True

        
    # Initialize LLM
    llm = Ollama(model="llama2", request_timeout=180.0)

    print("# embed_model", datetime.now())
    # Initialize HuggingFace Embedding Model for Vectorization
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Set the global tokenizer to use the tokenizer from HuggingFace for encoding inputs
    set_global_tokenizer(
        AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
    )

    print("# load data", datetime.now())
    # Load documents from the data directory into the Vector Store Index
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    # Create Vector Store Index with HuggingFace Embedding
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    # Create Prompt Template for Text-based Q&A
    chat_text_qa_msgs = [
    (
        "user",
        """You are a Q&A assistant. For all other inquiries, your main goal is to provide answers as accurately as possible, based on the instructions and context you have been given. If a question does not match the provided context or is outside the scope of the document, kindly advise the user to ask questions within the context of the document.
        Context:
        {context_str}
        Question:
        {query_str}
        """
    )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

    # Initialize Chat Memory Buffer for Conversation Memory
    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

    # Create Query Engine with LLM and Template
    query_engine = index.as_query_engine(
        llm=llm, 
        text_qa_template=text_qa_template, 
        streaming=True,
        memory=memory
        )
    
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = query_engine


    print("# loaded", datetime.now())

# Function to handle queries
def handle_query(query):
    streaming_response = st.session_state.query_engine.query(query)
    for text in streaming_response.response_gen:
        yield text

print("-- check", datetime.now())

# ============== Streamlit App ===============
st.title("POC LLM RAG ✅")
st.markdown("Retrieval-Augmented Generation (RAG) with Large Language Model (LLM) using llama-index library and Ollama.") 
st.markdown("start chat ...🚀")

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', "content": 'Hello! Ask me anything about the documents.'}]

# Sidebar to list documents
with st.sidebar:
    st.title("Documents:")
    docs = SimpleDirectoryReader(DATA_DIR).list_resources()
    for d in docs:
        file_name = str(d).split('/')[-1]
        st.info(file_name)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask me anything about the documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = handle_query(prompt)
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

