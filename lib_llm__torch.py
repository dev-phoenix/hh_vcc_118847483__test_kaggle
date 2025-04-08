# lib_llm__torch.py
'''
based on article:
source: https://habr.com/ru/articles/775870/
'''

import torch # (797.1 MB)
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from lib_model_names import selectModelName

# Leaderboard
# https://russiansuperglue.com/leaderboard/2

# https://huggingface.co/IlyaGusev/saiga_mistral_7b_lora
MODEL_NAME = "IlyaGusev/saiga_mistral_7b_lora"
# loaded. Error by runtime
# peft/peft_model.py", line 526, in <dictcomp>
#     "safetensors_file": index[p]["safetensors_file"],
# KeyError: 'safetensors_file'

# MODEL_NAME=model_name

MODEL_NAME, test_result, llm_result_is = selectModelName(-1)


def test_llm__torch():
    device = torch.device('cpu')
    # Загружаем модель
    print('='*30, 'Model')
    print(f'Model name: {MODEL_NAME}')
    print('='*30, 'get conf')
    config = PeftConfig.from_pretrained(MODEL_NAME)
    print(config)
    print('='*30, 'auto model')
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16,
        # device_map="auto",
        # device_map="balanced",
        device_map="cpu",
        offload_folder="offload"
    )
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to('cpu')

    print('='*30, 'PeftModel')
    model = PeftModel.from_pretrained(
        model,
        MODEL_NAME,
        torch_dtype=torch.float16
    )
    print('='*30, 'model.eval')
    model.eval()

    print('='*30, 'Next ...')
    # Определяем токенайзер
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    print('='*30, 'GenerationConfig ...')
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)

    print('='*30, 'generate ...')
    # Функция для обработки запросов
    def generate(model, tokenizer, prompt, generation_config):
        print('='*30, 'tokenizer ...')
        data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(model.device) for k, v in data.items()}
        print(data)
        print('='*30, 'model.generate ...')
        output_ids = model.generate(
            **data,
            generation_config=generation_config
        )[0]
        print(output_ids)
        print('='*30, 'output_ids ...')
        output_ids = output_ids[len(data["input_ids"][0]):]
        print('='*30, 'tokenizer.decode ...')
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(output)
        print('='*30, 'output ...')
        return output.strip()

    # Формируем запрос
    PROMT_TEMPLATE = '<s>system\nТы — Сайга, русскоязычный автоматический ассистент. \
Ты разговариваешь с людьми и помогаешь им.</s><s>user\n{inp}</s><s>bot\n'
    PROMT_TEMPLATE = '<s>system\n{hello}</s><s>user\n{inp}</s><s>bot\n'
    inp = 'Какое расстояние до Луны?'
    hello = 'Ты — русскоязычный ассистент.'
    inp = 'Привет'
    prompt = PROMT_TEMPLATE.format(inp=inp)

    # Отправляем запрос в llm
    output = generate(model, tokenizer, prompt, generation_config)

    print(output)