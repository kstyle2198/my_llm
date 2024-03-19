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

  
    file_path = "D:\AA_develop\my_llm\images\eng_receipt.png"
    img = Image.open(file_path)
    st.image(img, width=300)
    
    receipt_texts, receipt_boxes = paddle_scan(paddleocr,file_path)
    st.markdown(f"{receipt_texts}")
    st.markdown(f"{receipt_boxes}")

    draw_boundary_box(receipt_boxes)

    # if st.button("OCR--->Json"):
    #     import torch
    #     from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig

        # quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
        # bnb_config = BitsAndBytesConfig(
        #     llm_int8_enable_fp32_cpu_offload=True,
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )

        # bnb_config = BitsAndBytesConfig(
        #         load_in_4bit= True,
        #         bnb_4bit_quant_type= "nf4",
        #         bnb_4bit_compute_dtype= torch.float16,
        #         bnb_4bit_use_double_quant= False,
        #         )
        # control model memory allocation between devices for low GPU resource (0,cpu)
        # device_map = {
        #     "transformer.word_embeddings": 0,
        #     "transformer.word_embeddings_layernorm": 0,
        #     "lm_head": 0,
        #     "transformer.h": 0,
        #     "transformer.ln_f": 0,
        #     "model.embed_tokens": 0,
        #     "model.layers":0,
        #     "model.norm":0    
        # }
        # device = "cuda" if torch.cuda.is_available() else "cpu"

        # # model use for inference
        # model_id="mychen76/mistral7b_ocr_to_json_v1"
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_id, 
        #     trust_remote_code=True,  
        #     torch_dtype=torch.float16,
        #     # quantization_config=bnb_config,
        #     device_map=device_map)
        # # tokenizer
        # tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


        # prompt=f"""### Instruction:
        # You are POS receipt data expert, parse, detect, recognize and convert following receipt OCR image result into structure receipt data object. 
        # Don't make up value not in the Input. Output must be a well-formed JSON object.```json

        # ### Input:
        # {receipt_boxes}

        # ### Output:
        # """

        # with torch.inference_mode():
        #     inputs = tokenizer(prompt,return_tensors="pt",truncation=True).to(device)
        #     outputs = model.generate(**inputs, max_new_tokens=512) ##use_cache=True, do_sample=True,temperature=0.1, top_p=0.95
        #     result_text = tokenizer.batch_decode(outputs)[0]
        #     st.markdown(result_text)


