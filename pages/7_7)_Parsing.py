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


    uploadfile = st.file_uploader("PDF 업로드")
    try:
        with open(os.path.join('D:/AA_develop/my_llm/sample_pdf/',uploadfile.name),"wb") as f:
            f.write(uploadfile.getbuffer())
        pdf_path1 = f"D:/AA_develop/my_llm/sample_pdf/{uploadfile.name}"
    except:
        pdf_path1 = "D:\AA_develop\my_llm\sample_pdf\FWG.pdf"
    
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
    st.subheader("Extract Highlights(Annotations)")
   
    # Open the PDF
    doc = fitz.open(pdf_path1)

    # Define the RGB values for your colors
    PINK = (0.9686269760131836, 0.6000000238418579, 0.8196079730987549)
    YELLOW = (1.0, 0.9411770105361938, 0.4000000059604645)
    GREEN = (0.49019598960876465, 0.9411770105361938, 0.4000000059604645)
    RED = (0.9215689897537231, 0.2862749993801117, 0.2862749993801117)

    color_definitions = {"Pink": PINK, "Yellow": YELLOW, "Green": GREEN, "Red": RED}

    # Create separate lists for each color
    data_by_color = {"Pink": [], "Yellow": [], "Green": [], "Red": []}

    # Loop through every page
    for i in range(len(doc)):
        page = doc[i]
        annotations = page.annots()
        for annotation in annotations:
            if annotation.type[1] == 'Highlight':
                color = annotation.colors['stroke']  # Returns a RGB tuple
                if color in color_definitions.values():
                    # Get the detailed structure of the page
                    structure = page.get_text("dict")

                    # Extract highlighted text line by line
                    content = []
                    for block in structure["blocks"]:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                r = fitz.Rect(span["bbox"])
                                if r.intersects(annotation.rect):
                                    content.append(span["text"])
                    
                    content = " ".join(content)

                    # Append the content to the appropriate color list
                    for color_name, color_rgb in color_definitions.items():
                        if color == color_rgb:
                            data_by_color[color_name].append(content)
    
    data_by_color

    st.markdown("---")
    st.subheader("Extract Colored Texts")
    def flags_decomposer(flags):
        """Make font flags human readable."""
        l = []
        if flags & 2 ** 0:
            l.append("superscript")
        if flags & 2 ** 1:
            l.append("italic")
        if flags & 2 ** 2:
            l.append("serifed")
        else:
            l.append("sans")
        if flags & 2 ** 3:
            l.append("monospaced")
        else:
            l.append("proportional")
        if flags & 2 ** 4:
            l.append("bold")
        return ", ".join(l)



    # page = doc[0]
    results = []
    for page in doc:
    # read page text as a dictionary, suppressing extra spaces in CJK fonts
        blocks = page.get_text("dict", flags=11)["blocks"]
        for b in blocks:  # iterate through the text blocks
            for l in b["lines"]:  # iterate through the text lines
                for s in l["spans"]:  # iterate through the text spans
                    font_properties = "Font: '%s' (%s), size %g, color #%06x" % (
                        s["font"],  # font name
                        flags_decomposer(s["flags"]),  # readable font flags
                        s["size"],  # font size
                        s["color"],  # font color
                    )
                    if s["color"] != 0:
                        results.append(s["text"])
                    # st.markdown(f"Text: {s['text']}, color: {s['color']}")  # simple print of text
                    # st.markdown(font_properties)
    results








