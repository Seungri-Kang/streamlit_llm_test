from flask import Flask, jsonify
from flask import request
import gpu_main as main
import time
import transformers
import torch
import os
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
import gpu_model as gpu_model

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def hello_world():
    return 'Hello, World!'
    
@app.route('/chatbot', methods=['POST'])
def chatbot():
    global present_selected_model

    tempJson = request.get_json()
    present_selected_model = tempJson['selected_model']
    prompt = tempJson['prompt']

    tempJson = request.get_json()
    selected_model = tempJson['selected_model']
    prompt = tempJson['prompt']
    chatbots = main.ChatBot()
    # Llama-3-Korean
    if selected_model == 'Llama-3-Korean-8B':
        start = time.time()
        msg = chatbots.chat_llama3_8B(prompt)
        print(f"Llama-3-Korean-8B {time.time()-start:.4f} sec") 
        return msg
    elif selected_model == 'Llama-3-Koream-70B':
        msg = chatbots.chat_llama3_70B(prompt)
        return msg
    elif selected_model == 'Llama-3-Koream-8B-Quant':
        start = time.time()
        msg = chatbots.chat_llama3_8B_quant(prompt)
        print(f"Llama-3-Korean-8B-Quant {time.time()-start:.4f} sec") 
        return msg
    elif selected_model == 'EEVE-Korean':
        start = time.time()
        msg = chatbots.eeve_korean(prompt)
        print(f"EEVE-Korean {time.time()-start:.4f} sec") 
        return msg
    
    
@app.route('/summary', methods=['POST'])
def summary():
    tempJson = request.get_json()
    selected_model = tempJson['selected_model']
    prompt = tempJson['prompt']
    summarys = main.Summary()
    # Llama-3-Korean
    if selected_model == 'Llama-3-Korean-8B':
        start = time.time()
        msg = summarys.summary_llama3_8B(prompt)
        print(f"Llama-3-Korean-8B {time.time()-start:.4f} sec") 
        return msg
    elif selected_model == 'Llama-3-Koream-70B':
        msg = summarys.chat_llama3_70B(prompt)
        return msg
    elif selected_model == 'Llama-3-Koream-8B-Quant':
        start = time.time()
        msg = summarys.chat_llama3_8B_quant(prompt)
        print(f"Llama-3-Korean-8B-Quant {time.time()-start:.4f} sec") 
        return msg
    elif selected_model == 'gemma-Korean-7B':
        start = time.time()
        msg = summarys.summary_gemma_7B(prompt)
        print(f"gemma-Korean-7B {time.time()-start:.4f} sec") 
        return msg

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5656, debug=True)  # GPU, debug=True
