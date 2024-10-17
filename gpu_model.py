import transformers
import torch
import os
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)



def load_llama3_8B():
    model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16,
        device_map=0,
    )

    pipeline.model.eval()
    return pipeline, pipeline.tokenizer


def load_llama3_8B_quant():
    model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                quantization_config=bnb_config,
                                                device_map=0
                                                )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, add_special_tokens=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map={"":0},
    )

    return pipeline, tokenizer


def load_llama3_70B():
    model_name = "Bllossom/llama-3-Korean-Bllossom-70B"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                quantization_config=bnb_config,
                                                device_map=0
                                                )

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_special_tokens=True)


    return model, tokenizer



def load_eeve_korean():
    model = AutoModelForCausalLM.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0", quantization_config=bnb_config, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0")

    return model, tokenizer


def load_gemma_korean():
    model_id = "rtzr/ko-gemma-2-9b-it"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    pipeline.model.eval()
    return pipeline, pipeline.tokenizer