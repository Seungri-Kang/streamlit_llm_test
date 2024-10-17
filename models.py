from transformers import AutoTokenizer, AutoModelForCausalLM, TFGPT2LMHeadModel
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, BartForConditionalGeneration
import transformers
import time
import datetime
import time
import datetime
import torch
from openai import OpenAI
import models

def load_eeve():
    model_name = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer
    # # 입력 텍스트를 토큰화
    # input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # # 모델을 사용하여 텍스트 생성
    # output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=3, early_stopping=True)
    
    # # 생성된 텍스트 디코딩
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # # 응답 출력
    # return generated_text

def load_skt():
    model_name = "skt/kogpt2-base-v2"
    # 모델과 토크나이저 로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>')
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer
    # # 입력 텍스트를 토큰화
    # input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # # 모델을 사용하여 텍스트 생성
    # output = model.generate(input_ids, max_length=512, num_return_sequences=1, no_repeat_ngram_size=3, early_stopping=True)
    
    # # 생성된 텍스트 디코딩
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # return generated_text

def load_skt_finetuning():
    model_path = "./LLM_models/fine-tuning-model_hyundai"
    # 모델과 토크나이저 로드
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path, bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<unused0>')
    # 모델을 평가 모드로 전환
    model.eval()
    return model, tokenizer
    # # 디바이스 설정 (가능한 경우 GPU, 그렇지 않으면 CPU 사용)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    
    # # 입력 텍스트를 토큰화
    # input_ids = tokenizer.encode(f"<usr> {prompt} <sys>", return_tensors="pt").to(device)
    
    # with torch.no_grad():
    #     output = model.generate(input_ids, max_length=150, pad_token_id=tokenizer.eos_token_id)
    
    # response = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(response)
    # return response.split("<sys>")[-1].strip()

def load_kobart():
    model_name = "EbanLee/kobart-summary-v3"
    # 모델과 토크나이저 로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    # 모델을 평가 모드로 전환
    model.eval()
    return model, tokenizer

def load_skt_hyundai():
    model_path = "./LLM_models/kogpt2-chatbot-model_re-fine-tuning"
    # 모델과 토크나이저 로드
    model = TFGPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 모델을 평가 모드로 전환
    return model, tokenizer



def openai_model():
    print('openAI model')    