import streamlit as st
import base64


st.set_page_config(layout='wide', page_title="My_LLM")





if __name__ == "__main__":
    st.title("My LLM")
    st.markdown("***from OCR, Parsing, Embedding to LLM Inference***")
    st.markdown("---")


    file_ = open("C:/my_develop2/my_llm/images/aaa.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )