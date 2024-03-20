import streamlit as st
import pandas as pd
import cv2
from paddleocr import PaddleOCR, draw_ocr
from ast import literal_eval
import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
os.chdir("D:/AA_develop/my_llm")

st.set_page_config(layout="wide",page_title="OCR")




paddleocr = PaddleOCR(lang="en",ocr_version="PP-OCRv4",show_log = False,use_gpu=True)

def paddle_scan(paddleocr,img_path_or_nparray):
    result = paddleocr.ocr(img_path_or_nparray,cls=True)
    result = result[0]
    boxes = [line[0] for line in result]       #boundign box 
    txts = [line[1][0] for line in result]     #raw text
    scores = [line[1][1] for line in result]   # scores
    return  txts, result

def draw_boundary_box(result):
    res = result
    # st.markdown(type(res))
    boxes = [res[i][0] for i in range(len(result))] 
    # st.markdown(boxes)
    texts = [res[i][1][0] for i in range(len(result))]
    # st.markdown(texts)
    scores = [float(res[i][1][1]) for i in range(len(result))]

    img = cv2.imread(file_path) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    plt.figure(figsize=(65,65))
    annotated = draw_ocr(img, boxes, texts, scores, font_path="D:/AA_develop/my_llm/malgun.ttf") 
    st.image(annotated)



if __name__ == "__main__":
    st.title("OCR with paddleocr")

  
    file_path =  "D:\AA_develop\my_llm\images\eng_receipt.png"
    img = Image.open(file_path)
    st.image(img, width=300)
    
    receipt_texts, receipt_boxes = paddle_scan(paddleocr,file_path)
    st.markdown(f"{receipt_texts}")
    st.markdown(f"{receipt_boxes}")

    draw_boundary_box(receipt_boxes)


st.subheader("취소선 표시 텍스트 제거 방법 연구")

import PyPDF2
from pdfminer.high_level import extract_text
from pytesseract import image_to_string
from PIL import Image

def extract_text_from_pdf_with_cancellation(pdf_path):
    # Extract text using PyPDF2
    pdf_text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            pdf_text += page.extractText()

    # Extract text using pdfminer
    pdf_text += extract_text(pdf_path)

    # Extract text from images using pytesseract
    pdf_images = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        pdf_reader

        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            page
            
            if '/XObject' in page['/Resources']:
                xObject = page['/Resources']['/XObject'].getObject()
                for obj in xObject:
                    if xObject[obj]['/Subtype'] == '/Image':
                        size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                        data = xObject[obj].getData()
                        if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                            mode = "RGB"
                        else:
                            mode = "P"

                        if '/Filter' in xObject[obj]:
                            if xObject[obj]['/Filter'] == '/FlateDecode':
                                img = Image.frombytes(mode, size, data)
                                pdf_images.append(img)
                            elif xObject[obj]['/Filter'] == '/DCTDecode':
                                img = open(obj[1:] + ".jpg", "wb")
                                img.write(data)
                                img.close()
                                img = Image.open(obj[1:] + ".jpg")
                                pdf_images.append(img)
                            elif xObject[obj]['/Filter'] == '/JPXDecode':
                                img = open(obj[1:] + ".jp2", "wb")
                                img.write(data)
                                img.close()
                                img = Image.open(obj[1:] + ".jp2")
                                pdf_images.append(img)
    pdf_images
    for image in pdf_images:
        image_text = image_to_string(image)
        pdf_text += image_text

    # 취소선이 있는 텍스트 제거
    lines = pdf_text.split('\n')
    filtered_lines = [line for line in lines if not any(char == '-' for char in line)]
    filtered_text = '\n'.join(filtered_lines)

    # Remove text with strikethrough
    # You may need to implement your own logic to detect and remove strikethrough text

    return filtered_text

# Example usage
pdf_path = "D:\\AA_develop\\my_llm\\sample_pdf\\FWG.pdf"
extracted_text = extract_text_from_pdf_with_cancellation(pdf_path)
st.markdown(extracted_text)