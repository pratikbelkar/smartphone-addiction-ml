from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Smartphone Usage Risk',page_icon='📱',layout='centered')
st.title('📱 Smartphone Usage Risk')
st.caption('College ML demonstration trained on synthetic data. This is not a clinical assessment.')
model_file=ROOT/'artifacts'/'model.joblib';metadata_file=ROOT/'artifacts'/'metadata.json'
if not model_file.exists() or not metadata_file.exists():
    st.error('Train first: run `python train.py` in the project folder.');st.stop()
@st.cache_resource
def load():return joblib.load(model_file),json.loads(metadata_file.read_text())
model,meta=load()
with st.form('predict'):
    screen=st.slider('Daily screen time (hours)',0.0,16.0,6.0,.1)
    social=st.slider('Social media (hours)',0.0,float(screen),min(2.0,float(screen)),.1)
    gaming=st.slider('Gaming (hours)',0.0,float(max(screen-social,0.0)),min(1.0,float(max(screen-social,0.0))),.1)
    checks=st.slider('Phone checks per day',0,300,90)
    night=st.slider('Night phone use (minutes)',0,240,45)
    sleep=st.slider('Sleep per night (hours)',0.0,12.0,7.0,.1)
    study=st.slider('Study or work per day (hours)',0.0,12.0,4.0,.1)
    submitted=st.form_submit_button('Predict risk',type='primary')
if submitted:
    row=pd.DataFrame([dict(zip(meta['features'],[screen,social,gaming,checks,night,sleep,study]))])
    label=model.predict(row)[0]
    probs=dict(zip(model.classes_,model.predict_proba(row)[0]))
    st.subheader(f'Predicted category: {label}')
    st.bar_chart(pd.DataFrame({'Estimated probability':probs}))
    st.info('These probabilities reflect the synthetic training data. They are not validated health risk percentages.')
