import streamlit as st
import pandas as pd
import random
import PyPDF2
from loguru import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader



if "document1" not in st.session_state:
    st.session_state.document1 = []

if "text" not in st.session_state:
    st.session_state.text = []




if __name__ == "__main__":

    st.title("Document Loaders & Metadata Edit")

    upload_files = st.file_uploader("File Uploader - PDF, DOCX, TXT, CSV, PPTX")
    # upload_files
    # st.session_state.document1 = get_text(upload_files)
    # st.subheader("Origin Document Data")
    # st.session_state.document1
    
    
    filename = upload_files.name
    file_path = "D:\\AA_develop\\my_llm\\sample_pdf\\" + filename
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    pages
    len(pages)
    

    # # try:
    # filename = upload_files.name
    # file_path = "D:\\AA_develop\\my_llm\\sample_pdf\\" + filename
    # pdf_file = open(file_path, 'rb')
    # reader = PyPDF2.PdfReader(pdf_file)

    # page_num = 0
    # page = reader.pages[page_num]
    # st.session_state.text = page.extract_text()
    # st.session_state.text

    print('--------------------------------------------------')
        
    # except:
    #     st.empty()

    st.markdown("---")
    st.subheader("Metadata Edit Test")

    new_doc = pages
    # 메타데이터에 추가할 값들  
    fruits = ["Apple", "Banana", "Strawberry", "Carrot"]

    origin_meta = []
    for i in range(len(new_doc)):
        origin_meta.append(new_doc[i].metadata)

    
    new_meta = origin_meta
    for idx, meta in enumerate(new_meta):
        fruit = random.choice(fruits)
        meta["fruit"] = fruit

    st.text_area("##### Origin Metadata", origin_meta)
    st.text_area("##### New Metadata - (fruit 값 추가)", new_meta)


    st.subheader("New Document Data")
    new_doc
    




