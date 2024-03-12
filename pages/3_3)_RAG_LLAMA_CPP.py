import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

import httpcore
setattr(httpcore, 'SyncHTTPTransport', "AsyncHTTPProxy")

import os
import re
import time
import pandas as pd
import streamlit as st
# from langchain.document_loaders import PyPDFLoader
# from langchain.document_loaders import Docx2txtLoader
# from langchain.document_loaders import UnstructuredPowerPointLoader
# from loguru import logger

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

from langchain.callbacks import StreamlitCallbackHandler
from llama_cpp import Llama

st.set_page_config(layout="wide",page_title="RAG_CHATBOT")


####### Session State Variables ####################################


if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if 'questions' not in st.session_state:
    st.session_state['questions'] = list()

if 'answer' not in st.session_state:
    st.session_state['answer'] = list()

if 'trans_answer' not in st.session_state:
    st.session_state['trans_answer'] = list()

if 'src_docu1' not in st.session_state:
    st.session_state['src_docu1'] = list()

if 'src_meta1' not in st.session_state:
    st.session_state['src_meta1'] = list()

if 'src_docu2' not in st.session_state:
    st.session_state['src_docu2'] = list()

if 'src_meta2' not in st.session_state:
    st.session_state['src_meta2'] = list()

if 'response' not in st.session_state:
    st.session_state['response'] = list()

if 'text_splitter' not in st.session_state:
    st.session_state['text_splitter'] = ""

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = ""

if 'llm' not in st.session_state:
    st.session_state['llm'] = ""

if 'dbqa' not in st.session_state:
    st.session_state['dbqa'] = ""

if 'replied' not in st.session_state:
    st.session_state['replied'] = ""


#### functions ###########################################################

from datetime import datetime, timedelta
def calculate_time_delta(start_time, end_time):
    # Calculate the time difference (time delta) in seconds
    time_difference = end_time - start_time
    seconds = time_difference.seconds
    return seconds


from googletrans import Translator
class Google_Translator:
    def __init__(self):
        self.translator = Translator()
        self.result = {'src_text': '', 'src_lang': '', 'tgt_text': '', 'tgt_lang': ''}

    def translate(self, text, lang='ko'):
        translated = self.translator.translate(text, dest=lang)
        self.result['src_text'] = translated.origin
        self.result['src_lang'] = translated.src
        self.result['tgt_text'] = translated.text
        self.result['tgt_lang'] = translated.dest

        return self.result

    def translate_file(self, file_path, lang='ko'):
        with open(file_path, 'r') as f:
            text = f.read()
        return self.translate(text, lang)

def trans(en):
    translator = Google_Translator()
    result = translator.translate(str(en))
    if "tgt_text" in result.keys():
        return result["tgt_text"]
    else:
        return result["src_text"]


sample_text = '''
Korea logged a current account surplus for the ninth consecutive month in January as exports for chips and cars continued to rebound, though the growth of the positive balance slowed from the previous month.
 
The country logged a surplus of $3.05 billion in January, slowing from $7.41 billion the previous month, data from the Bank of Korea (BOK) showed on Friday.
 
The goods account racked up a $4.24 billion surplus in the first month of 2024, following $8.04 billion the previous month.


 
Exports jumped 14.7 percent in January from a year earlier to $55.52 billion, led by a 52.8 percent surge in chips and 24.8 percent growth in automobiles, while vessels soared 75.8 percent over the same period.
 
Chip exports recovered on the back of server memory chips as demand started to pick up from the second half of last year, according to the central bank. Increasing demand from China and a rise in semiconductor prices propelled the growth.
 
Outbound shipments to the United States rose 27.1 percent to $10.24 billion, while those to China jumped 16 percent to $10.68 billion.
 
Imports declined 8.1 percent over the same period to $50.98 billion, driven by a 11.3 percent drop in inbound shipments of raw materials.
 
The primary income account, which tracks the wages of foreign workers, dividend payments from overseas and interest income, registered a $1.62 billion surplus in January, slowing from $2.46 billion a month earlier.
 
The services account deficit expanded to $2.66 billion from a deficit of $2.54 billion in the cited period, driven by the $1.47 billion deficit in the travel account with an increase in overseas travelers.
 
“January is usually a month when overseas travel grows in time for winter vacation season,” said a spokesperson for the BOK. “The travel account is expected to improve in February as the number of incoming passengers rose on China’s Lunar New Year holiday.”
 
The current account surplus is expected to expand in February as the trade surplus increased in the month, the central bank said. The current account balance is projected to stay in positive territory based on the goods account in the first half of the year, and the surplus trend will grow more evident with its expansion in the second half of the year.
 
Last year, the country reported a current account surplus of $35.49 billion, which was higher than the central bank’s estimate of $30 billion.
 
The BOK forecast the current account surplus to widen to $52 billion this year.
'''

if __name__ == "__main__":

    st.title("📑 :red[MY RAG CHATBOT] with :blue[LLAMA2] & :green[MISTRAL]")
    st.markdown("---")

    texts = st.text_area("🍀 Context", sample_text, height=500)

    col1, col2, col3, col4, col5 = st.columns(5)

    models = {"🦙LLAMA2_7B": "C:/my_develop2/my_llm/model/llama-2-7b-chat.ggmlv3.q8_0.bin",
            "🪁MISTRAL_7B": "C:/my_develop2/my_llm/model/mistral-7b-instruct-v0.1.Q8_0.gguf",}

    model_types = {"🦙LLAMA2_7B": "llama",
                   "🪁MISTRAL_7B":"mistral",}

    col701, col702, col703 = st.columns([3,4,4])
    with col701:
        sel01 = st.selectbox("🚩 **:red[Select LLM]**", ("🦙LLAMA2_7B", "🪁MISTRAL_7B"))
        LLM_model = models[sel01]
        model_type = model_types[sel01]

    with col702:
        temp_value = st.slider("🌡️ **Temperature**", min_value=0.0, max_value=2.0, value=0.0, step=0.1)

    with col703:
        max_token = st.slider("🌡️ **Max_New_Token**", min_value=256, max_value=2000, value=256, step=100)

    btn111 = st.button("⚙️ Load LLM", type='primary')
    with st.spinner("Loading..."):
        try:
            if btn111==True:
                st.session_state['text_splitter'] = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
                docs = st.session_state['text_splitter'].split_text(texts)

                with col1:
                    st.info("🍉text_splitter Completed")

                st.session_state['embeddings'] = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                                    model_kwargs={'device':'cpu'},)

                with col2:
                    st.info("🍊embeddings Completed")

                vectorstore = FAISS.from_texts(docs, st.session_state['embeddings'])

                with col3:
                    st.info("🍎vectorstore Completed")
                    
                st.session_state['llm'] = CTransformers(model=LLM_model, # Location of downloaded GGML model
                                                        model_type=model_type,
                                                        stream=True,
                                                        config={'max_new_tokens': max_token,
                                                                'temperature': temp_value,
                                                                'context_length': 4096})

                with col4:
                    st.info("🍓LLM Loading Completed")


                qa_template = """Use the following pieces of information to answer the user's question.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.

                Context: {context}
                Question: {question}

                Only return the helpful answer below and nothing else.
                Helpful answer:
                """

                prompt = PromptTemplate(template=qa_template, input_variables=['context', 'question'])
                st_callback = StreamlitCallbackHandler(st.container())
                dbqa = RetrievalQA.from_chain_type(llm=st.session_state['llm'],
                                                    chain_type='stuff',
                                                    callbacks=[st_callback],
                                                    retriever= vectorstore.as_retriever(search_type="mmr", search_kwargs={'k':2}),
                                                    return_source_documents=True,
                                                    chain_type_kwargs={'prompt': prompt})
                st.session_state['dbqa'] = dbqa
                with col5:
                    st.info("🍇RetrievalQA Chain Completed")

            else:
                st.empty()
        except:
            st.error("🚨 Add or Insert the Knowledge Context")
    

    input100 = st.text_area("🖊️ **Input your Question**")

    chk1 = st.checkbox("Translation into Korean Language", value=True)
    st.session_state['replied'] = st.button("⚙️ Submit", type='primary')


    with st.spinner("🤗 inference..."):
        if st.session_state['replied']:
            st.session_state['questions'].append(input100)
            start_time = datetime.now()

            response = st.session_state['dbqa']({'query': input100})
            # response

            st.session_state['answer'].append(response["result"])
            st.session_state['src_docu1'].append(response["source_documents"][0].page_content)
            # st.session_state['src_meta1'].append(response["source_documents"][0].metadata["source"])
            try:
                st.session_state['src_docu2'].append(response["source_documents"][1].page_content)
            #     st.session_state['src_meta2'].append(response["source_documents"][1].metadata["source"])
            except:
                st.session_state['src_docu2'].append("")
            #     st.session_state['src_meta2'].append("")

            st.markdown(f"😆 :blue[{st.session_state['answer'][-1]}]")
            end_time = datetime.now()
            delta = calculate_time_delta(start_time, end_time)
            st.warning(f"⏱️ Answering Time Delta(sec) : {delta}")

            try:
                if chk1:
                    st.session_state['trans_answer'].append(trans(st.session_state['answer'][-1]))
                    st.markdown(f"😃[번역] :blue[{st.session_state['trans_answer'][-1]}]")
                else:
                    st.session_state['trans_answer'].append("")
            except:
                st.session_state['trans_answer'].append("")

            end_time = datetime.now()
            delta = calculate_time_delta(start_time, end_time)
            st.warning(f"⏱️ Answering Time Delta(sec) after Translation : {delta}")

        with st.expander("✔️ **Collection of Answers(Download)**", expanded=True):
            df = pd.DataFrame({
                "Question":st.session_state['questions'],
                "Answer":st.session_state['answer'],
                "Trans_Answer":st.session_state['trans_answer'],
                "Evidence1":st.session_state['src_docu1'],
                # "근거파일1":st.session_state['src_meta1'],
                "Evidence2":st.session_state['src_docu2'],
                # "근거파일2":st.session_state['src_meta2']
                })
            st.dataframe(df, use_container_width=True)


            @st.cache_data
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8-sig')

            csv = convert_df(df)

            st.download_button(
                label="🗄️ Download data as CSV",
                data=csv,
                file_name='answers.csv',
                mime='text/csv',
                )

