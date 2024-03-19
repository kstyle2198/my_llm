import streamlit as st
import pandas as pd
import pdfplumber
from spire.pdf.common import *
from spire.pdf import *
import fitz


st.set_page_config(layout="wide",page_title="Parsing")

def block_based_parsing_by_page(pdf_path, page_num):
    results = ""
    with pdfplumber.open(pdf_path) as pdf:
        # Iterate through each page
        # for page in pdf.pages:
        # Extract words using .extract_words() method
        page = pdf.pages[page_num]
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
    bounding_box = (3, 70, 590, 770)   #default : (0, 0, 595, 841)
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


    pdf_path1 = "D:\AA_develop\my_llm\sample_pdf\FWG.pdf"
    # pdf_path1 = "D:\AA_develop\my_llm\sample_pdf\LNGC.pdf.pdf"

    page_num = st.number_input("페이지 번호 입력(시범 34)", step=1)


    st.subheader("PDF 문서 내용 및 표 추출")



    if st.button("block_based_parsing"):
        st.session_state['blocked_content'] = block_based_parsing_by_page(pdf_path1, page_num)
    
    st.text_area("block_based_parsing", st.session_state['blocked_content'])

    tb_num = st.number_input("테이블번호입력",min_value=0, max_value=10, value="min")
    if st.button("extract_table"):
        st.session_state['extract_table'] = extract_table(pdf_path1, page_num, tb_num)
        st.session_state['markdown_content'] = table_converter(st.session_state['extract_table'])
    
    st.text_area("extract_table(페이지의 첫번째 테이블만 조회중)", st.session_state['extract_table'])   
    st.text_area("Markdown Type - table_string_convert", st.session_state['markdown_content'])   

    st.markdown("---")
    st.subheader("PDF 문서 Cropping 및 도면 등 이미지 추출")


    if st.button("Cropping"):
        st.session_state['cropped_result'] = cropping(pdf_path1, int(page_num))
    
    st.text_area("Cropped Result - bounding_box = (3, 70, 590, 770)", st.session_state['cropped_result'])

    if st.button("샘플페이지 이미지추출"):
        st.session_state['image_filenames'] = image_extractor(pdf_path1, int(page_num))

    for image_path in st.session_state['image_filenames']:
        st.image(image_path)



    st.markdown("---")
    st.subheader("Extract Highlights")
    pg_num = st.number_input("페이지입력",min_value=0, max_value=10, value="min")
    doc = fitz.open(pdf_path1)
    st.markdown(f"총페이지수: {len(doc)}")
    page = doc[pg_num]
    page
    # list to store the co-ordinates of all highlights
    highlights = []
    # loop till we have highlight annotation in the page
    annot = page.first_annot
    while annot:
        if annot.type[0] == 8:
            all_coordinates = annot.vertices
            if len(all_coordinates) == 4:
                highlight_coord = fitz.Quad(all_coordinates).rect
                highlights.append(highlight_coord)
            else:
                all_coordinates = [all_coordinates[x:x+4] for x in range(0, len(all_coordinates), 4)]
                for i in range(0,len(all_coordinates)):
                    coord = fitz.Quad(all_coordinates[i]).rect
                    highlights.append(coord)
        annot = annot.next
    
    highlights

    all_words = page.get_text_words()
    st.markdown("x0, y0, x1, y1, “word”, block_no, line_no, word_no")
    all_words

    # List to store all the highlighted texts
    highlight_text = []
    for h in highlights:
        sentence = [w[4] for w in all_words if   fitz.Rect(w[0:4]).intersects(h)]
        highlight_text.append(" ".join(sentence))

    highlight_text

    total_highlight_text = " ".join(highlight_text)
    st.markdown(f"total_highlight_text: {total_highlight_text}")


    # st.markdown("---")
    # doc = fitz.open(pdf_path1)

    # # List to store all the highlighted texts
    # highlight_text = []

    # # loop through each page
    # for page in doc:

    #     # list to store the co-ordinates of all highlights
    #     highlights = []
        
    #     # loop till we have highlight annotation in the page
    #     annot = page.first_annot
    #     while annot:
    #         if annot.type[0] == 8:
    #             all_coordinates = annot.vertices
    #             if len(all_coordinates) == 4:   
    #                 highlight_coord = fitz.Quad(all_coordinates).rect
    #                 highlights.append(highlight_coord)
    #             else:
    #                 all_coordinates = [all_coordinates[x:x+4] for x in range(0, len(all_coordinates), 4)]
    #                 for i in range(0,len(all_coordinates)):
    #                     coord = fitz.Quad(all_coordinates[i]).rect
    #                     highlights.append(coord)
    #         annot = annot.next
            
    #     all_words = page.get_text("words")
    #     total_highlight_text = ""
    #     for h in highlights:
    #         sentence = [w[4] for w in all_words if fitz.Rect(w[0:4]).intersects(h)]
    #         highlight_text.append(" ".join(sentence))
    #         total_highlight_text = " ".join(highlight_text)
    # st.markdown(f"total_highlight_text: {total_highlight_text}")
    






    









