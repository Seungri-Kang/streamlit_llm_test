import streamlit as st
from openai import OpenAI
import models
import torch
import tensorflow as tf
import requests
import json

st.title("💬 Chatbot")

def extract_model_names(models_info: list) -> tuple:
    return tuple(model for model in models_info)

server = "CPU"
on = st.sidebar.toggle("GPU")
if on:
    server = "GPU"

@st.cache_resource
def load_model(selected_model):
    if selected_model == 'skt':
        model, tokenizer = models.load_skt()
    elif selected_model == 'eeve':
        model, tokenizer = models.load_eeve()
    elif selected_model == 'skt-fine tuning':
        model, tokenizer = models.load_skt_finetuning()
    elif selected_model == 'openai':
        model, tokenizer = ("","")
    elif selected_model == 'skt-hyundai':
        model, tokenizer = models.load_skt_hyundai()
    else:
        pass
    return model, tokenizer

# CPU
if server == "CPU":
    models_info = ['skt','eeve','openai','skt-fine tuning','skt-hyundai']
    available_models = extract_model_names(models_info)
    if available_models:
        selected_model = st.selectbox(
            "Pick a model available locally on your system ↓", available_models
        )
        print(selected_model)
        try:
            model, tokenizer = load_model(selected_model)
        except:
            pass
    else:
        st.warning("You have not pulled any model yet!", icon="⚠️")


    if selected_model == 'openai':
        with st.sidebar:
            openai_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
            "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
            "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"


    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # Select openai
        if selected_model == 'openai':
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
                st.stop()
            
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(model="gpt-4o-mini", messages=st.session_state.messages)
            msg = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
        
        #Select skt
        elif selected_model == 'skt':
            input_text = f"당신은 챗봇입니다. 질문에 대한 적절한 답변을 출력하세요. ### 질문: {prompt} ### 답변:"
            input_ids = tokenizer.encode(input_text, return_tensors='pt')
            output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=3, early_stopping=True)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            msg = response.split("### 답변:")[-1].strip()
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
        
        # Select eeve
        elif selected_model == 'eeve':
            input_text = f"당신은 챗봇입니다. 질문에 대한 적절한 답변을 출력하세요. ### 질문: {prompt} ### 답변:"
            input_ids = tokenizer.encode(input_text, return_tensors='pt')
            output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=3, early_stopping=True)
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            print(response)
            msg = response.split("### 답변:")[-1].strip()
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
        
        elif selected_model == 'skt-fine tuning':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            input_ids = tokenizer.encode(f"<usr> {prompt} <sys>", return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(input_ids, max_length=150, pad_token_id=tokenizer.eos_token_id)
            
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            response = response.split("<sys>")[-1].strip()
            msg = response[len(prompt)+1:]
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
         
        elif selected_model == 'skt-hyundai':
            sent = '<usr>' + prompt + '<sys>'
            input_ids = [tokenizer.bos_token_id] + tokenizer.encode(sent)
            input_ids = tf.convert_to_tensor([input_ids])
            output = model.generate(input_ids, max_length=100, do_sample=True, top_k=20)
            sentence = tokenizer.decode(output[0].numpy().tolist())
            msg = sentence.split('<sys> ')[1].replace('</s>', '')    
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
        else:
            st.warning("You have not pulled any model yet!", icon="⚠️")

# GPU
else:
    models_info = ['Llama-3-Korean-8B', 'Llama-3-Korean-70B','Llama-3-Koream-8B-Quant', 'EEVE-Korean']
    available_models = extract_model_names(models_info)
    if available_models:
        selected_model = st.selectbox(
            "Pick a model available locally on your system ↓", available_models
        )
        print(selected_model)
        try:
            model, tokenizer = load_model(selected_model)
        except:
            pass
    else:
        st.warning("You have not pulled any model yet!", icon="⚠️")


    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        header = {
            'Content-type' : 'application/json'
        }
        params = {'selected_model' : selected_model, 'prompt': prompt}
        url = 'http://19.19.20.178:5656/chatbot'
        
        if selected_model == 'Llama-3-Korean-8B':
            response = requests.post(url=url, data=json.dumps(params), headers=header)
            msg = response.text
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
        elif selected_model == 'Llama-3-Koream-8B-Quant':
            response = requests.post(url=url, data=json.dumps(params), headers=header)
            msg = response.text
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
        elif selected_model == 'EEVE-Korean':
            response = requests.post(url=url, data=json.dumps(params), headers=header)
            msg = response.text
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)