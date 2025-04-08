# lib_model_names.py

defmodel = 18
def smnGen(mod_num=-1, getinfo=True, with_result=False):
    if getinfo == True:
        for mod_num in range(1, defmodel+1):
            r = selectModelName(mod_num, False, True)
            # print(r)
            model_name, test_result, llm_result_is = r 
            yield model_name, test_result, llm_result_is 
        return '','',''

def selectModelName(mod_num=-1, getinfo=False, with_result=False):
    # print(':', mod_num, getinfo, with_result)
    '''
    local python version 3.8
    its hase not 'case' or multiprocessor
    '''
    defmodel = 17
    defmodel = 15 # сайг мистраль лора 7 миллиардов
    if mod_num == -1:
        mod_num = defmodel

    MODEL_NAME=''
    test_result=''
    llm_result_is=False

    # first tested with lib_llm_torch.py
    if mod_num == 1:
        # https://huggingface.co/IlyaGusev/saiga_mistral_7b_lora
        MODEL_NAME = "IlyaGusev/saiga_mistral_7b_lora"
        llm_result_is=False
        test_result='''
        # with test with lib peft
        # loaded. Error by runtime
        # peft/peft_model.py", line 526, in <dictcomp>
        #     "safetensors_file": index[p]["safetensors_file"],
        # KeyError: 'safetensors_file'
        '''

    if mod_num == 2:
        # https://huggingface.co/Defetya/qwen-4B-saiga
        MODEL_NAME = "Defetya/qwen-4B-saiga"
        llm_result_is=False
        test_result='''
        # config error
        # ValueError: Can't find 'adapter_config.json' at 'Defetya/qwen-4B-saiga'
        '''

    if mod_num == 3:
        # https://huggingface.co/ai-forever/rugpt3xl
        MODEL_NAME = "ai-forever/rugpt3xl"
        llm_result_is=False
        test_result='''
        # config error
        # requests.exceptions.HTTPError: 404 Client Error: 
        # Not Found for url: https://huggingface.co/ai-forever/rugpt3xl/resolve/main/adapter_config.json
        '''

    if mod_num == 4:
        # https://huggingface.co/IlyaGusev/saiga_13b_lora
        MODEL_NAME = "IlyaGusev/saiga_13b_lora"
        llm_result_is=False
        test_result='''
        # loaded
        #  warnings.warn(f'for {key}: copying from a non-meta parameter in the checkpoint to a meta '
        # /python3.8/site-packages/torch/nn/modules/module.py:2068: 
        # UserWarning: for base_model.model.model.layers.39.self_attn.o_proj.lora_B.default.weight:
        # copying from a non-meta parameter in the checkpoint to a meta parameter in the current model, 
        # which is a no-op. 
        # (Did you mean to pass `assign=True` to assign items in the state dictionary 
        # to their corresponding key in the module instead of copying them in place?)

        # python3.8/site-packages/torch/cuda/__init__.py:128:
        # UserWarning: CUDA initialization: 
        # The NVIDIA driver on your system is too old (found version 11000). 
        # Please update your GPU driver by downloading and installing a new version 
        # from the URL: http://www.nvidia.com/Download/index.aspx 
        # Alternatively, go to: https://pytorch.org to install a PyTorch version 
        # that has been compiled with your version of the CUDA driver. 
        # (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)

        # from langchain version
        # MODEL_NAME='lmsys/vicuna-13b-v1.5-16k'
        # work for langchain version
        # not work for peft version

        # at this moment, summ of weights is 59G
        '''

    if mod_num == 5:
        # at this moment, summ of weights is 59G
        MODEL_NAME = 'sanchit-gandhi/whisper-small-ru-1k-steps'
        llm_result_is=False
        llm_result_is=True # ?
        test_result='''
        loaded but not applicable
        https://huggingface.co/sanchit-gandhi/whisper-small-ru-1k-steps
        В общем, её тренировали на аудио фразах,
на абхазском )))
https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/\
viewer/ab/train?p=210&views%5B%5D=ab_train
        '''

    model_name = MODEL_NAME


    # first tested with lib_llm_torch.py
    if mod_num == 6:
        model_name = "deepseek-ai/deepseek-llm-67b-base"
        llm_result_is=False
        test_result='''
        # fail not found model part number 3 of 17
        # loaded 19G
        '''

    if mod_num == 7:
        # https://huggingface.co/microsoft/Phi-3-medium-128k-instruct
        model_name = "microsoft/Phi-3-medium-128k-instruct"
        test_result='''
        # time to load 1h 10m
        # loaded 27G
        # fail: 
        # File "/python3.8/site-packages/transformers/generation/configuration_utils.py", 
        # line 567, in validate
        #     if self.pad_token_id is not None and self.pad_token_id < 0:
        # TypeError: '<' not supported between instances of 'list' and 'int'
        '''

    if mod_num == 8:
        model_name = 'bambucha/saiga-llama3'
        llm_result_is=False
        test_result='''
        # filed
        # OSError: bambucha/saiga-llama3 is not a local folder and is not a valid model identifier 
        # listed on 'https://huggingface.co/models'
        # If this is a private repository, make sure to pass a token having permission 
        # to this repo either by logging in with `huggingface-cli login` 
        # or by passing `token=<your_token>`
        '''

    if mod_num == 9:
        # https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/ab/train?p=210&views%5B%5D=ab_train
        # https://huggingface.co/sanchit-gandhi/whisper-small-ru-1k-steps
        model_name = 'sanchit-gandhi/whisper-small-ru-1k-steps'
        llm_result_is=False
        test_result='''
        # loaded : 967M
        # time to load: 5m
        # first error: 
        # ValueError: You have explicitly specified `forced_decoder_ids`. 
        # Please remove the `forced_decoder_ids` argument in favour of `input_ids` 
        # or `decoder_input_ids` respectively.
        # Uncknown result. Force CPU to heigher level and nothing heppened throw 40 minutes.
        # result is unclear.
        # run after 1h 12m of thinking
        # next run after 3m
        # model was treined on audio on abhasian language

        # last output:
        #   warnings.warn(
        # The attention mask is not set and cannot be inferred from input 
        # because pad token is same as eos token. 
        # As a consequence, you may observe unexpected behavior. 
        # Please pass your input's `attention_mask` to obtain reliable results.

        # first run Time forced: 01h 12m 22s
        # second run Time forced: 00h 02m 13s
        '''

    if mod_num == 10:
        # https://huggingface.co/yandex/YandexGPT-5-Lite-8B-instruct-GGUF
        model_name = 'yandex/YandexGPT-5-Lite-8B-instruct-GGUF'
        llm_result_is=False
        test_result='''
too huge
        '''

    if mod_num == 11:
        # https://huggingface.co/sberbank-ai/rugpt3small_based_on_gpt2
        model_name = 'sberbank-ai/rugpt3small_based_on_gpt2'
        llm_result_is=False
        test_result='''
        # OSError: sberbank-ai/rugpt3small_based_on_gpt2 does not appear 
        # to have a file named generation_config.json. 
        # Checkout 'https://huggingface.co/sberbank-ai/rugpt3small_based_on_gpt2/tree/main'
        # for available files.
        '''

    if mod_num == 12:
        # https://platform.openai.com/docs/models/gpt-3-5
        model_name = 'docs/models/gpt-3-5'
        llm_result_is=False
        test_result='''
        # some trables with config
        '''

    if mod_num == 13 and False:
        print('Woodoo coming')
        llm_result_is=False
        exit()

    if mod_num == 14:
        # https://huggingface.co/mistralai/Mistral-7B-v0.1
        model_name = 'mistralai/Mistral-7B-v0.1'
        llm_result_is=False
        test_result='''
        # Cannot access gated repo for url
        #  https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/config.json.
        # Access to model mistralai/Mistral-7B-v0.1 is restricted. 
        # You must have access to it and be authenticated to access it. Please log in.
        '''



    if mod_num == 15:
        # https://huggingface.co/IlyaGusev/saiga_mistral_7b_lora
        model_name = 'IlyaGusev/saiga_mistral_7b_lora'
        llm_result_is=True
        test_result='''
        # loaded: 36m
        # AttributeError: forced_decoder_ids (maybe not found)
        # first result: 1h 20m
        # second: 2h 48m
        # third: 04h 44m 07s
        # query and answers:


        # first result: 1h 20m
system
Урҭ еиҿахәыҭхәыҭуан. user
Урҭ еиҿахәыҭхәыҭуан. bot
Теперь я понимаю, что это слово можно перевести как "поддерживать друг друга".
Time forced: 01h 22m 58s

        # second: 2h 48m
Time forced: 02h 48m 27s
system
Ты — Сайга, русскоязычный автоматический ассистент. user
Раскажи о себе. Что умеешь делать? bot
Я - искусственный интеллект, разработанный для общения с людьми и выполнения различных задач. Я могу:

1. Разговаривать на естественном языке: я понимаю и отвечаю на вопросы, которые вы можете мне задать.
2. Исполнять команды: я могу выполнять различные
Time forced: 02h 48m 27s
        
        # third: 04h 44m 07s
Time forced: 04h 44m 07s
system
Ты — Сайга, русскоязычный автоматический ассистент. user
Mожешь дать статистику на основании данных из файла csv? Что тебе, для этого нужно? bot
Конечно, я могу это сделать. Для начала вам необходимо предоставить мне файл CSV или его URL-адрес. Файл CSV должен содержать данные в виде таблицы, где каждая строка представляет одну запись, а каждый столбец - одно поле данных.

После полу
Time forced: 04h 44m 07s

        # next ...
Time forced: 00h 01m 02s
<s>system
Ты — русскоязычный ассистент.</s><s>user
Привет.</s><s>bot

Time forced: 00h 01m 02s
Time forced: 00h 44m 29s
tensor([    1,  1587,    13, 28875, 28829,  1040, 24379,  1789, 28811,  7763,
        28826,  4086,  1622,  6993,   608,  6089, 28723,     2,     1,  2188,
           13, 28847,   892,  8496, 28723,     2,     1, 12435,    13, 28847,
          892,  8496, 28808,  5471, 28795,  4025,  5378, 21589, 28826, 28822,
        28804,     2])
Time forced: 00h 44m 29s
Time forced: 00h 44m 29s
system
Ты — русскоязычный ассистент. user
Привет. bot
Привет! Как могу помочь?
Time forced: 00h 44m 29s

        '''


    if mod_num == 16:
        # https://github.com/RussianNLP/morocco
        # https://huggingface.co/russiannlp/rugpt3-small
        model_name = 'russiannlp/rugpt3-small'
        llm_result_is=False
        test_result='''
        # notfound
        '''

    if mod_num == 17:
        # https://huggingface.co/openai-community/gpt2
        model_name = 'openai-community/gpt2'
        llm_result_is=True
        test_result='''
        # small ~ 548M
        # and fast ~ 5 minutes to answer
        # but return some insane answer.
        # only english
        '''

    if mod_num == 18:
        # https://huggingface.co/zlsl/ru_lora_large_startrek
        model_name = 'zlsl/ru_lora_large_startrek'
        llm_result_is=False
        test_result='''
        # small ~ 548M
        # raise ValueError(
ValueError: Unrecognized model in zlsl/ru_lora_large_startrek. 
Should have a `model_type` key in its config.json,
or contain one of the following strings in its name: albert, align,.
...
        '''
    # print('+'*30)
    if with_result:
        return model_name, test_result, llm_result_is,
    return model_name, test_result, llm_result_is
    # print('ext')