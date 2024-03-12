import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.embeddings import HuggingFaceEmbeddings
import time
from dotenv import load_dotenv
from loguru import logger

load_dotenv()  #

groq_api_key = os.environ['GROQ_API_KEY']


@st.cache_data
def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        # elif '.docx' in doc.name:
        #     loader = Docx2txtLoader(file_name)
        #     documents = loader.load_and_split()
        # elif '.pptx' in doc.name:
        #     loader = UnstructuredPowerPointLoader(file_name)
        #     documents = loader.load_and_split()
        doc_list.extend(documents)
    return doc_list



if "docs" not in st.session_state:
    st.session_state.docs = ""

if "text_splitter" not in st.session_state:
    st.session_state.text_splitter = ""

if "documents" not in st.session_state:
    st.session_state.documents = ""

if "embeddings" not in st.session_state:
    st.session_state.embeddings = ""

if "vector" not in st.session_state:
    st.session_state.vector = ""

if "prompt" not in st.session_state:
    st.session_state.prompt = ""





if __name__ == "__main__":
    st.title("Chat with Docs - Groq Edition :) ")
    st.markdown("---")

    uploadfile = st.file_uploader("Upload File")
    uploadfile
    try: 
        st.session_state.docs = get_text([uploadfile])
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.documents = st.session_state.text_splitter.split_documents( st.session_state.docs)
        st.text_area("Attached Context", st.session_state.docs)

        st.session_state.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                                model_kwargs={'device':'cpu'},)

        st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

        llm = ChatGroq(
                        groq_api_key=groq_api_key, 
                        model_name='mixtral-8x7b-32768'
                )

        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context. 
        Think step by step before providing a detailed answer. 
        I will tip you $200 if the user finds the answer helpful. 
        <context>
        {context}
        </context>

        Question: {input}""")

        document_chain = create_stuff_documents_chain(llm, prompt)

        retriever = st.session_state.vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        st.session_state.prompt = st.text_input("Input your prompt here")


        # If the user hits enter
        if prompt:
            # Then pass the prompt to the LLM
            start = time.process_time()
            response = retrieval_chain.invoke({"input": st.session_state.prompt})
            print(f"Response time: {time.process_time() - start}")

            st.write(response["answer"])

            # With a streamlit expander
            with st.expander("Document Similarity Search"):
                # Find the relevant chunks
                for i, doc in enumerate(response["context"]):
                    # print(doc)
                    # st.write(f"Source Document # {i+1} : {doc.metadata['source'].split('/')[-1]}")
                    st.write(doc.page_content)










    except:
        st.empty()


    