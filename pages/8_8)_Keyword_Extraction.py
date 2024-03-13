import streamlit as st
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline
import re

st.set_page_config(layout="wide",page_title="RAG_CHATBOT")


if "model" not in st.session_state:
    st.session_state.model =""

if "tokenizer" not in st.session_state:
    st.session_state.tokenizer =""

if "generator" not in st.session_state:
    st.session_state.generator =""

if "res_list" not in st.session_state:
    st.session_state.res_list =[]

if "keywords_list" not in st.session_state:
    st.session_state.keywords_list =[]

if "cleaned_keyword_list" not in st.session_state:
    st.session_state.cleaned_keyword_list =[]



def extract_keywords_only(text):
    # Find the last occurrence of "[/INST]"
    last_inst_index = text.rfind("[/INST]")

    # Extract the portion of the text after the last occurrence of "[/INST]"
    last_inst_text = text[last_inst_index + len("[/INST]"):]

    # Split the text by spaces and commas to extract words
    words = last_inst_text.split()

    # Cleaning up the extracted words
    cleaned_words = [word.strip(",") for word in words]

    # 키워드 리스트에서 숫자 제거
    cleaned_words_without_num = [word for word in cleaned_words if not any(char.isdigit() for char in word)]

    # 키워드 리스트에서 특수문자포함 단어 제거 "(a)"
    cleaned_words_without_spe_char = [word for word in cleaned_words_without_num if not re.match(r'^\(\w\)$', word)]

    return cleaned_words_without_spe_char

    


if __name__ == "__main__":
    st.title("Keyword Extraction with LLM")
    st.markdown("---")

    with st.spinner("Loading..."):
        if st.button("Load Model(Mistral)"):
            # Set gpu_layers to the number of layers to offload to GPU. 
            # Set to 0 if no GPU acceleration is available on your system.
            st.session_state.model = AutoModelForCausalLM.from_pretrained(
                "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                model_type="mistral",
                gpu_layers=0,
                context_length=4096,
                hf=True)
            
            # Tokenizer
            st.session_state.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

            # Pipeline
            st.session_state.generator = pipeline(
                model=st.session_state.model, tokenizer=st.session_state.tokenizer,
                task='text-generation',
                max_new_tokens=50,
                repetition_penalty=1.1
                )
            st.info("Model Loaded")
    st.markdown("---")
    st.subheader("Basic Chatbot")
    q = st.text_input("Your Question: ")
    if st.button("Submit"):
        response = st.session_state.generator(q)
        st.session_state.res_list.append(response)
    st.text_area("Response Collection", st.session_state.res_list, key="wuroe")
    st.session_state.res_list



    st.markdown("---")
    st.subheader("Keyword Extraction with LLM(Mistral-7b)")

    target_text = st.text_area("Target_text")
    example_prompt = """
                    <s>[INST]
                    Please give me the keywords that are present in this document and separate them with commas.
                    Make sure you to only return the keywords and say nothing else. For example, don't say:
                    "Here are the keywords present in the document"
                    [/INST] meat, beef, eat, eating, emissions, steak, food, health, processed, chicken</s>"""
    
    keyword_prompt = f"""
                    [INST]
                    I have the following document:
                    - {target_text}

                    Please give me the keywords that are present in this document and separate them with commas.
                    Make sure you to only return the keywords and say nothing else. 
                    Do not repeat prompt content in your answer. Just return only keywords.
                    [/INST]
                    """
    full_prompt = example_prompt + keyword_prompt
    prompt = st.text_area("Full Combined Prompt", full_prompt, key="wiee")


    if st.button("Extract"):
        response = st.session_state.generator(prompt)
        cleaned_words = extract_keywords_only(response[0]["generated_text"])

        st.session_state.keywords_list.append(response)
        st.session_state.cleaned_keyword_list.append(cleaned_words)
    st.text_area("Full Answers", st.session_state.keywords_list, key="dkshfl")
    
    # cleaned_words = extract_keywords_only(st.session_state.keywords_list[-1][0]["generated_text"])
    # st.session_state.cleaned_keyword_list.append(cleaned_words)

    st.text_area("Cleaned Keywords Collection", st.session_state.cleaned_keyword_list, key="dkshfsssl")

    



    