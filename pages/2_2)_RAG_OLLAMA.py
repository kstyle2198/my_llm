
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFLoader

import os
import tempfile
import streamlit as st
from streamlit_chat import message


class ChatCSV:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        """
        Initializes the question-answering system with default configurations.
        This constructor sets up the following components:
        - A ChatOllama model for generating responses ('neural-chat').
        - A RecursiveCharacterTextSplitter for splitting text into chunks.
        - A PromptTemplate for constructing prompts with placeholders for question and context.
        """
        # Initialize the ChatOllama model with 'neural-chat'.
        self.model = ChatOllama(model="neural-chat")

        # Initialize the RecursiveCharacterTextSplitter with specific chunk settings.
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

        # Initialize the PromptTemplate with a predefined template for constructing prompts.
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are a helpful HR assistant that analyses resumes from different candidates.
            Use the following pieces of retrieved context to answer the question.
            Give names when possible. If you don't know the answer,
            just say that you don't know.  [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )
        
    def ingest(self, csv_file_path: str):
        '''
        Ingests data from a CSV file containing resumes, process the data, and set up the
        components for further analysis.
        Parameters:
        - csv_file_path (str): The file path to the CSV file.
        Usage:
        obj.ingest("/path/to/data.csv")
        This function uses a CSVLoader to load the data from the specified CSV file.
        Args:
        - file.path (str): The path to the CSV file.
        - encoding (str): The character encoding of the file (default is 'utf-8').
        - source_column (str): The column in the CSV containing the data (default is "Resume").
        '''        
        loader = CSVLoader(
            file_path=csv_file_path,
            encoding='utf-8',
            source_column="Resume"
            )
        
        # loads the data
        data = loader.load()
        data

        # splits the documents into chunks
        chunks = self.text_splitter.split_documents(data)
        chunks = filter_complex_metadata(chunks)

        # creates a vector store using embedding
        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        # sets up the retriever
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        # Define a processing chain for handling a question-answer scenario.
        # The chain consists of the following components:
        # 1. "context" from the retriever
        # 2. A passthrough for the "question"
        # 3. Processing with the "prompt"
        # 4. Interaction with the "model"
        # 5. Parsing the output using the "StrOutputParser"
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        
    def ask(self, query: str):
        """
        Asks a question using the configured processing chain.
        Parameters:
        - query (str): The question to be asked.
        Returns:
        - str: The result of processing the question through the configured chain.
        If the processing chain is not set up (empty), a message is returned
        prompting to add a CSV document first.
        """
        if not self.chain:
            return "Please, add a CSV document first."

        return self.chain.invoke(query)

    def clear(self):
        """
        Clears the components in the question-answering system.
        This method resets the vector store, retriever, and processing chain to None,
        effectively clearing the existing configuration.
        """
        # Set the vector store to None.
        self.vector_store = None

        # Set the retriever to None.
        self.retriever = None

        # Set the processing chain to None.
        self.chain = None



class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="llama2:7b")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None




def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    st.header("ChatPDF")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

if __name__ == "__main__":
    # Call the "page" function to set up and run the Streamlit app.
    page()