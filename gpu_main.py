import gpu_model as models
import time
import transformers
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import torch

class ChatBot():
    def __init__(self) -> None:
        pass

    def chat_llama3_8B(self, prompt):
        start = time.time()
        pipeline, pipeline.tokenizer = models.load_llama3_8B()
        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
        messages = [
            {"role": "system", "content": f"{PROMPT}"},
            {"role": "user", "content": f"{prompt}"}
            ]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,  # 속도 개선을 위해 샘플링 범위 조정 top_p=0.9
            num_return_sequences=1,  # 한 번에 하나의 답변만 생성
            pad_token_id=pipeline.tokenizer.pad_token_id  # 패딩 토큰 설정
        )
        # 응답 출력
        msg = outputs[0]["generated_text"][len(prompt):]
        return msg

    def chat_llama3_8B_quant(self, prompt):
        pipeline, tokenizer = models.load_llama3_8B_quant()
        pipeline.tokenizer.pad_token = tokenizer.eos_token
        pipeline.tokenizer.padding_side = 'right'


        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
        messages = [
            {"role": "system", "content": f"{PROMPT}"},
            {"role": "user", "content": f"{prompt}"}
            ]

        
        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,  # 속도 개선을 위해 샘플링 범위 조정 top_p=0.9
            num_return_sequences=1,  # 한 번에 하나의 답변만 생성
            pad_token_id=pipeline.tokenizer.pad_token_id  # 패딩 토큰 설정
        )

        # 응답 출력
        msg = outputs[0]["generated_text"][len(prompt):]
        return msg

    def chat_llama3_70B(self, prompt):
        pipeline, tokenizer = models.load_llama3_70B()
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'


        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
        messages = [
            {"role": "system", "content": f"{PROMPT}"},
            {"role": "user", "content": f"{prompt}"}
            ]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,  # 속도 개선을 위해 샘플링 범위 조정 top_p=0.9
            num_return_sequences=1,  # 한 번에 하나의 답변만 생성
            pad_token_id=pipeline.tokenizer.pad_token_id  # 패딩 토큰 설정
        )

        # 응답 출력
        msg = outputs[0]["generated_text"][len(prompt):]
        return msg
    
    def eeve_korean(self, text):
        model, tokenizer  = models.load_eeve_korean()
        prompt_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: {prompt}\nAssistant:\n"

        # prompt = prompt_template.format(prompt=prompt)
        # text = '한국의 수도는 어디인가요? 아래 선택지 중 골라주세요.\n\n(A) 경성\n(B) 부산\n(C) 평양\n(D) 서울\n(E) 전주'
        model_inputs = tokenizer(prompt_template.format(prompt=text), return_tensors='pt')

        outputs = model.generate(**model_inputs, max_new_tokens=256)
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # print(output_text)
        response = output_text.split('Assistant:')[1]
        return response

class Summary():
    def __init__(self) -> None:
        pass

    def summary_llama3_8B(self,prompt):
        pipeline, pipeline.tokenizer = models.load_llama3_8B()
        PROMPT = '''아래 고객과 상담원 간의 통화 내역을 요약해 주세요. 주요 논의 사항, 고객의 요청 및 상담원의 해결 방안을 중심으로 간결하게 정리해 주세요.'''
        messages = [
            {"role": "system", "content": f"{PROMPT}"},
            {"role": "user", "content": f"{prompt}"}
            ]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,  # 속도 개선을 위해 샘플링 범위 조정 top_p=0.9
            num_return_sequences=1,  # 한 번에 하나의 답변만 생성
            pad_token_id=pipeline.tokenizer.pad_token_id  # 패딩 토큰 설정
        )

        # 응답 출력
        msg = outputs[0]["generated_text"][len(prompt):]
        return msg

    def summary_gemma_7B(self,prompt):
        pipeline, pipeline.tokenizer = models.load_gemma_korean()
        PROMPT = '''아래 고객과 상담원 간의 통화 내역을 요약해 주세요. 주요 논의 사항, 고객의 요청 및 상담원의 해결 방안을 중심으로 간결하게 정리해 주세요.'''
        messages = [
            {"role": "system", "content": f"{PROMPT}"},
            {"role": "user", "content": f"{prompt}"}
            ]

        # instruction = "서울의 유명한 관광 코스를 만들어줄래?"

        # messages = [
        #     {"role": "user", "content": f"{instruction}"}
        # ]

        prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        return outputs[0]["generated_text"][len(prompt):]
