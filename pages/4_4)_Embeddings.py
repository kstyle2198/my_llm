import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(layout="wide",page_title="Embeddings")



if 'model' not in st.session_state:
    st.session_state['model'] = ""

if 'embedding_result' not in st.session_state:
    st.session_state['embedding_result'] = ""

if 'decomp_embedding_result' not in st.session_state:
    st.session_state['decomp_embedding_result'] = ""

if 'full_cosine_sim_df' not in st.session_state:
    st.session_state['full_cosine_sim_df'] = pd.DataFrame()

if 'cosine_sim_df' not in st.session_state:
    st.session_state['cosine_sim_df'] = pd.DataFrame()

if 'decomp_full_cosine_sim_df' not in st.session_state:
    st.session_state['decomp_full_cosine_sim_df'] = pd.DataFrame()

if 'decomp_cosine_sim_df' not in st.session_state:
    st.session_state['decomp_cosine_sim_df'] = pd.DataFrame()

if 'chart_df' not in st.session_state:
    st.session_state['chart_df'] = pd.DataFrame()

if 'sel2' not in st.session_state:
    st.session_state['sel2'] = ""


samples = {
    "eng" : ['The team enjoyed the hike through the meadow',
            'The team enjoyed the hike through the mountains',
            'The team has not enjoyed the hike through the meadows',
            'The national park had great views',
            'There were lot of rare animals in national park',
            'Olive oil drizzled over pizza tastes delicious'],
    'eng1': ['FAN PANEL냉각 110V,2PH,SJ1238HA1/120120*38',
            'FAN PANEL냉각 AC220VV,1PH,SJ1238HA2,120*120*38',
            'FAN PANEL냉각 12V,2PH,HFB44B,COOLACE SEPA',
            '환풍기 HYE SUNG HV-35B',
            'NBK MEGA CHECK CLESNER 450ML'],
    'eng2': ['emergency', 'fire', 'vacation', 'hospital', 'baseball', 'injury', 'evacuation'],
    'eng3': ['''
            ||HEATING MEDIUM|COOLING MEDIUM|
            |Source|M/E jacket cooling F.W. (M/E model : HYUNDAI-MAN B&W 6G70ME-C10.5-HPSCR)|S.W.|
            |Flow rate (m3/h)|Maker's standard|Maker's standard|
            |℃ Inlet temp.( )|83|32|
            |Press.(bar)|4.5|Maker's standard|
            ||Distillate pump|Ejector pump|
            |Capacity|Maker's standard|To be supplied by shipyard|
            |Total head|Maker's standard|None|
            |Shaft seal|Maker's standard|None|
            ''',
            '''
            ||HEATING MEDIUM|COOLING MEDIUM|
            |Source|M/E jacket cooling F.W. (M/E model : HYUNDAI-MAN B&W 6G60ME-C10.5-HPSCR)|S.W.|
            |Flow rate (m3/h)|Maker's standard|Maker's standard|
            |℃ Inlet temp.( )|83|32|
            |Press.(bar)|5.5|Maker's standard|
            ||Distillate pump|Ejector pump|
            |Capacity|Maker's standard|To be supplied by shipyard|
            |Total head|Maker's standard|None|
            |Shaft seal|Maker's standard|None|
            ''',
            '''
            ||UNIT COOLER FOR ENGINE CONTROL ROOM|UNIT COOLER FOR SWITCHBOARD ROOM|UNIT COOLER FOR WORKSHOP|
            |Q’ty / Ship|Two(2) sets|Four(4) sets|One(1) set|
            |Cooling Capacity|15,000 kcal/h|60,000 kcal/h|15,000 kcal/h|
            |Refrigerant|R-407C|R-407C|R-407C|
            |Air Flow|Maker's standard|Maker's standard|Maker's standard|
            |Remark|#, ####|##|Without thermal insulation|

            '''
            ],

    "kor" : ['오늘까지 프리젠테이션 발표자료를 준비해야 합니다.',
            '어제 늦은 밤까지 자료 검토하느라 야근을 했습니다.',
            '오늘 미팅은 오전과 오후에 이어져 실시될 예정입니다.',
            '주말에는 에버랜드에 놀러갈 것입니다.',
            '남산 공원에는 많은 사람들이 봄의 정취를 느끼기 위해 나들이 중입니다.',
            '주말에는 축구경기를 관람할 예정입니다.'],
}



def draw_2d(df):
    fig = px.scatter(
        df,
        x=0,
        y=1,
        text = df.index
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

def draw_3d(df):
    fig = px.scatter_3d(df, x=0, y=1, z=2,text = df.index,)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


#######################################################################################
if __name__ == "__main__":

    st.title("Embedding Test")
    st.markdown("영어, 한글 문장들을 임베딩 모델에 입력후, 코사인 유사도 및 차원축소 시각화 실험 가능")
    st.markdown("---")

    sample_names = list(samples.keys())
    sel0 = st.selectbox("Select Sample List", sample_names)

    sentences = st.text_area("리스트 인풋 형태로...", samples[sel0])
    sentences =  eval(sentences)  # 'str'---> list

    embedding_models = ['paraphrase-MiniLM-L6-v2', 'jhgan/ko-sroberta-multitask']
    sel1 = st.selectbox("Select Embedding Model", embedding_models)
    btn1 = st.button("Load Model", type='primary')

    with st.spinner("Loading..."):
        if btn1:
            st.session_state['model'] = SentenceTransformer(f'{sel1}')
    st.session_state['model']
    


    col11, col22 = st.columns(2)
    with col11:
        btn33 = st.button("Embedding & Decompose_PCA", type='primary')
    with col22:
        sel3 = st.selectbox("Select Decomposed Dimension", [2, 3])


    with st.spinner("Embedding..."):
        if btn33:
            try:
                st.session_state['embedding_result']= st.session_state['model'] .encode(sentences)
                with st.expander(f"Embedding Result / {st.session_state['embedding_result'].shape}", expanded=False):
                    st.session_state['embedding_result']

                similarities = util.cos_sim(st.session_state['embedding_result'], st.session_state['embedding_result'])
                numpy_data = similarities.numpy()
                st.session_state['full_cosine_sim_df'] = pd.DataFrame(numpy_data)

                st.session_state['full_cosine_sim_df'].index = samples[sel0]
                st.session_state['full_cosine_sim_df'].columns = samples[sel0]


                PCA_model = PCA(n_components = sel3)
                PCA_model.fit(st.session_state['embedding_result'])
                st.session_state['decomp_embedding_result'] = PCA_model.transform(st.session_state['embedding_result'])
            
                decomp_similarities = util.cos_sim(st.session_state['decomp_embedding_result'], st.session_state['decomp_embedding_result'])

                decomp_numpy_data = decomp_similarities.numpy()
                st.session_state['decomp_full_cosine_sim_df'] = pd.DataFrame(decomp_numpy_data)
                st.session_state['decomp_full_cosine_sim_df'].index = samples[sel0]
                st.session_state['decomp_full_cosine_sim_df'].columns = samples[sel0]

                
                
                st.session_state['chart_df']  = pd.DataFrame(st.session_state['decomp_embedding_result'])
                st.session_state['chart_df'] .index = samples[sel0]


                if sel3 == 2:
                    draw_2d(st.session_state['chart_df'] )
                else:
                    draw_3d(st.session_state['chart_df'] )

            except:
                st.empty()
                
        try:
            st.session_state['sel2'] = st.selectbox("Select Target Sentence", samples[sel0])

            st.session_state['cosine_sim_df'] = st.session_state['full_cosine_sim_df'][[st.session_state['sel2']]]
            st.dataframe(st.session_state['cosine_sim_df'], use_container_width=True)

            st.session_state['decomp_cosine_sim_df'] = st.session_state['decomp_full_cosine_sim_df'][[st.session_state['sel2']]]
            st.dataframe(st.session_state['decomp_cosine_sim_df'], use_container_width=True)
        except:
            st.empty()



    # col11, col22 = st.columns(2)
    # with col11:
    #     btn33 = st.button("Decompose_PCA", type='primary')
    # with col22:
    #     sel3 = st.selectbox("Select Decomposed Dimension", [2, 3])
    # if btn33:
        # PCA_model = PCA(n_components = sel3)
        # PCA_model.fit(st.session_state['embedding_result'])
        # st.session_state['decomp_embedding_result'] = PCA_model.transform(st.session_state['embedding_result'])
    
        # decomp_similarities = util.cos_sim(st.session_state['decomp_embedding_result'], st.session_state['decomp_embedding_result'])

        # decomp_numpy_data = decomp_similarities.numpy()
        # st.session_state['decomp_full_cosine_sim_df'] = pd.DataFrame(decomp_numpy_data)
        # st.session_state['decomp_full_cosine_sim_df'].index = samples[sel0]
        # st.session_state['decomp_full_cosine_sim_df'].columns = samples[sel0]

        # sel4 = st.session_state['sel2']
        # st.session_state['decomp_cosine_sim_df'] = st.session_state['decomp_full_cosine_sim_df'][[sel4]]
        # with st.expander("Cosine Similarity Comparison :red[(after Decomposition)]", expanded=True):
        #     st.dataframe(st.session_state['decomp_cosine_sim_df'], use_container_width=True)

        
        # t_df = pd.DataFrame(st.session_state['decomp_embedding_result'])
        # t_df.index = samples[sel0]


        # if sel3 == 2:
        #     draw_2d(t_df)
        # else:
        #     draw_3d(t_df)





        # except:
        #     st.empty()

    # sel4 = sel3
    # st.session_state['decomp_cosine_sim_df'] = st.session_state['decomp_full_cosine_sim_df'][[sel2]]
    # with st.expander("Cosine Similarity Comparison :red[(after Decomposition)]", expanded=True):
    #     st.dataframe(st.session_state['decomp_cosine_sim_df'], use_container_width=True)

    
    # t_df = pd.DataFrame(st.session_state['decomp_embedding_result'])
    # t_df.index = samples[sel0]

    # try:
    #     if sel3 == 2:
    #         draw_2d(t_df)
    #     else:
    #         draw_3d(t_df)
    # except:
    #     pass