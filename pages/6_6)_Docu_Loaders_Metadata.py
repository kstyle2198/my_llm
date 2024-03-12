import streamlit as st
import pandas as pd
import random
from loguru import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader

@st.cache_data
def get_text(docs):
    doc_list = []
    for doc in docs:
        try:
            file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
            with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
                file.write(doc.getvalue())
                logger.info(f"Uploaded {file_name}")
            if '.pdf' in doc.name:
                loader = PyPDFLoader(file_name)
                documents = loader.load_and_split()
            elif '.docx' in doc.name:
                loader = Docx2txtLoader(file_name)
                documents = loader.load_and_split()
            elif '.pptx' in doc.name:
                loader = UnstructuredFileLoader(file_name)
                documents = loader.load_and_split()
            elif '.txt' in doc.name:
                loader = TextLoader(file_name)
                documents = loader.load_and_split()
            elif '.csv' in doc.name:
                loader = CSVLoader(file_name)
                documents = loader.load()
            doc_list.extend(documents)
        except:
            pass
    return doc_list


if "doc" not in st.session_state:
    st.session_state.doc = []

if "text" not in st.session_state:
    st.session_state.text = []




if __name__ == "__main__":

    st.title("Document Loaders & Metadata Edit")

    upload_files = st.file_uploader("File Uploader - PDF, DOCX, TXT, CSV, PPTX")
    st.session_state.doc = get_text([upload_files])
    st.subheader("Origin Document Data")
    st.session_state.doc
    
    

    import PyPDF2
    # filename = "/Users/Florian/Downloads/1706.03762.pdf"
    try:
        filename = upload_files.name
        pdf_file = open(filename, 'rb')
        reader = PyPDF2.PdfReader(pdf_file)

        page_num = 0
        page = reader.pages[page_num]
        st.session_state.text = page.extract_text()

        print('--------------------------------------------------')
        
    except:
        st.empty()

    st.markdown("---")
    st.subheader("Metadata Edit Test")

    new_doc = st.session_state.doc

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
    

    
    # pdf_file.close()



