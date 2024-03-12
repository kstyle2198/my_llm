import streamlit as st
import pandas as pd
import pdfplumber
from spire.pdf.common import *
from spire.pdf import *

st.set_page_config(layout="wide",page_title="Parsing")

def block_based_parsing(pdf_path):
    results = ""
    with pdfplumber.open(pdf_path) as pdf:
        # Iterate through each page
        for page in pdf.pages:
            # Extract words using .extract_words() method
            words = page.extract_words()
            lines = {}
            for word in words:
                line_top = word['top']
                if line_top not in lines:
                    lines[line_top] = []
                lines[line_top].append(word['text'])
            
            # Sort and print lines based on their y-coordinate
            for top in sorted(lines.keys()):
                result = ""
                if len(lines[top]) > 1:
                    result = ' '.join(lines[top])
                    # print(result)
                results = results + "\n" + result
    return results



def extract_table(pdf_path, page_num, table_num):
    # Open the pdf file
    pdf = pdfplumber.open(pdf_path)
    # Find the examined page
    table_page = pdf.pages[page_num]
    # Extract the appropriate table
    table = table_page.extract_tables()[table_num]
    return table


def table_converter(table):
    table_string = ''
    # Iterate through each row of the table
    for row_num in range(len(table)):
        row = table[row_num]
        # Remove the line breaker from the wrapped texts
        cleaned_row = [item.replace('\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item for item in row]
        # Convert the table into a string 
        table_string+=('|'+'|'.join(cleaned_row)+'|'+'\n')
    # Removing the last line break
    table_string = table_string[:-1]
    return table_string


def cropping(pdf_path, page_num):
    pdf = pdfplumber.open(pdf_path)
    pdf_page = pdf.pages[page_num]
    bounding_box = (3, 70, 590, 770)
    cropped_pdf = pdf_page.crop(bounding_box, relative=False, strict=True)
    cropped_result = cropped_pdf.extract_text()
    return cropped_result

def image_extractor(pdf_path, page_num):
    doc = PdfDocument()
    doc.LoadFromFile(pdf_path)
    page = doc.Pages[page_num]

    images = []
    for image in page.ExtractImages():
        images.append(image)
    index = 0
    image_filenames = []
    for image in images:
        imageFileName = 'Image-{0:d}.png'.format(index)
        image_filenames.append(imageFileName)
        index += 1
        image.Save(imageFileName, ImageFormat.get_Png())
    doc.Close()
    return image_filenames

    print("이미지 추출 저장 완료")


if 'blocked_content' not in st.session_state:
    st.session_state['blocked_content'] = ""

if 'extract_table' not in st.session_state:
    st.session_state['extract_table'] = ""

if 'markdown_content' not in st.session_state:
    st.session_state['markdown_content'] = ""

if 'cropped_result' not in st.session_state:
    st.session_state['cropped_result'] = ""

if 'image_filenames' not in st.session_state:
    st.session_state['image_filenames'] = []

if __name__ == "__main__":
    st.title("Parsing")
    st.markdown("---")

    pdf_path = "C:\my_develop2\my_llm\sample_pdf\c4611_sample_explain.pdf"
    st.subheader("PDF 문서 내용 및 표 추출")
    if st.button("block_based_parsing"):
        st.session_state['blocked_content'] = block_based_parsing(pdf_path)
    
    st.text_area("block_based_parsing", st.session_state['blocked_content'])

    if st.button("extract_table"):
        st.session_state['extract_table'] = extract_table(pdf_path, 0, 0)
        st.session_state['markdown_content'] = table_converter(st.session_state['extract_table'])
    
    st.text_area("extract_table", st.session_state['extract_table'])   
    st.text_area("Markdown Type - table_string_convert", st.session_state['markdown_content'])   

    st.markdown("---")
    st.subheader("PDF 문서 Cropping 및 도면 등 이미지 추출")
    pdf_path1 = "C:\my_develop2\my_llm\sample_pdf\LNGC.pdf.pdf"
    page_num = st.number_input("페이지 번호 입력(시범 77)", step=1)

    if st.button("Cropping"):
        st.session_state['cropped_result'] = cropping(pdf_path1, int(page_num))
    
    st.text_area("Cropped Result - bounding_box = (3, 70, 590, 770)", st.session_state['cropped_result'])

    if st.button("샘플페이지 이미지추출"):
        st.session_state['image_filenames'] = image_extractor(pdf_path1, int(page_num))

    for image_path in st.session_state['image_filenames']:
        st.image(image_path)





    









