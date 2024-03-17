import fasttext
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from seamless_communication.inference import Translator
import warnings

warnings.filterwarnings("ignore")

def get_lang_decoder():
    """
    This function loads the language identification model.
    """
    model_path = hf_hub_download(repo_id="AsphyXIA/baarat-hin-kan-en-decoder", filename="model_hin_en_kan_decode.bin")
    lang_decoder = fasttext.load_model(model_path)
    return lang_decoder

def get_dict_map():
    """
    This function returns the dictionary map for languages.
    """
    dict_map = {"__label__eng":"English",'__label__kan':"Kannada","__label__hindi":"Hindi"}
    return dict_map

def get_translation_prompt():
    """
    This function returns the translation prompt.
    """
    translation_prompt = """[INST] Below is an instruction that describes a task, paired with an input that provides further context. Write a response that
    appropriately completes the request.

    ### Instruction:
    Translate the following {} text to {}. Understand the context of the text and do not provide a raw translation.[/INST]
    [SEP]
    ### Input:
    {}
    [SEP]
    ### Response:
    {}"""
    return translation_prompt

def get_q_a_prompt():
    """
    This function returns the question answering prompt.
    """
    q_a_prompt = """[INST] Below is an instruction that describes a task, paired with an input that provides further context. Write a response that
    appropriately completes the request.

    ### Instruction:
    Answer the question based on the context provided below. If the context is empty, answer the question directly.[/INST]

    ### Context:
    {}

    ### Input:
    {}

    ### Response:
    {}"""
    return q_a_prompt

def get_prompt_template_inst_kannada():
    """
    This function returns the instruction prompt for Kannada.
    """
    prompt_template_inst_kannada = """
    [INST]Below is a conversation between a user and an assistant with an input in {} that describes an instruction that provides context for the user's
    requirement. Write a response in {} that appropriately completes the request.

    ### Instruction:
    {}[/INST]

    ### Response:
    {}"""
    return prompt_template_inst_kannada

def get_prompt_template_inst_hindi():
    """
    This function returns the instruction prompt for Hindi.
    """
    prompt_template_inst_hindi = """[INST] Below is a conversation between a user and an assistant that describes a task, that provides context for the user's
    requirement. Write a response that appropriately completes the request.

    ### Input:
    {}[/INST]

    ### Response:
    {}"""
    return prompt_template_inst_hindi

def get_prompt_template_inst_english():
    """
    This function returns the instruction prompt for English.
    """
    prompt_template_inst_english = """[INST] Below is an input that describes a task, paired with an input that provides further context. Write a response that
    appropriately completes the request.

    ### Input:
    {}[/INST]

    ### Response:
    {}"""
    return prompt_template_inst_english

def get_translator():
    """
    This function returns the translator object.
    """
    translator = Translator("seamlessM4T_medium", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:0"),dtype=torch.float16)
    return translator
