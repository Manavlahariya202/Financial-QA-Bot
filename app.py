import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import time
import tempfile

# Load environment variables
load_dotenv()

# Set API keys
groq_api_key = os.environ.get('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

# Streamlit App Configuration
st.title("Financial QA Bot with Llama2")
st.sidebar.title("Upload Financial PDF")

# Upload PDF
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if "vector" not in st.session_state:
    st.session_state.vector = None  # Initialize vector state

if uploaded_file:
    st.sidebar.success("PDF uploaded successfully!")

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load PDF content
    pdf_loader = PyPDFLoader(temp_file_path)
    documents = pdf_loader.load()

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    # Generate embeddings and store in FAISS vectorstore
    embeddings = OllamaEmbeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    st.session_state.vector = vector_store
    st.sidebar.success("Data indexed successfully!")

# Query Input
query = st.text_input("Ask a financial question:")

if query and st.session_state.vector:
    # Initialize Llama2 model
    llm = Ollama('llama2')

    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        You are a financial expert who extracts insights from financial tables in PDFs.
        Provide the most accurate response based on the financial question.

        <context>
        {context}
        <context>
        Questions: {input}
        """
    )

    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Get response
    start = time.process_time()
    response = retrieval_chain.invoke({"input": query})
    process_time = time.process_time() - start

    # Display the response
    st.write(f"### Question: {query}")
    st.write("### Response:")
    st.write(response['answer'])

    # Show similar document chunks
    with st.expander("Relevant Document Chunks"):
        for doc in response["context"]:
            st.write(doc.page_content)
            st.write("---")
    
    st.write(f"Response Time: {process_time:.2f} seconds")
else:
    if not st.session_state.vector:
        st.warning("Please upload a PDF to index data.")
