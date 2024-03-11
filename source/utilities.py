import fasttext
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from seamless_communication.inference import Translator
import warnings
warnings.filterwarnings("ignore")

"""--------------------------------------------------------------------
Utilty variables and imports used in the main file for processing
--------------------------------------------------------------------"""

""" 
load the language idenitifaction model
"""
model_path = hf_hub_download(repo_id="AsphyXIA/baarat-hin-kan-en-decoder", filename="model_hin_en_kan_decode.bin")
lang_decoder = fasttext.load_model(model_path)

dict_map = {"__label__eng":"English",'__label__kan':"Kannada","__label__hindi":"Hindi"}

### TRANSLATION TASK
translation_prompt = """[INST] Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Translate the following {} text to {}. Understand the context of the text and do not provide a raw translation.[/INST]
[SEP]
### Input:
{}
[SEP]
### Response:
{}"""
target = ""

### QUESTION ANSWERING
q_a_prompt = """[INST] Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the question based on the context provided below. If the context is empty, answer the question directly.[/INST]

### Context:
{}

### Input:
{}

### Response:
{}"""

context_eng = """"""

### INSTRUCTION KANNADA
prompt_template_inst_kannada = """
[INST]Below is a conversation between a user and an assistant with an input in {} that describes an instruction that provides context for the user's requirement. Write a response in {} that appropriately completes the request.

### Instruction:
{}[/INST]

### Response:
{}"""

### INSTRUCTION HINDI
prompt_template_inst_hindi = """[INST] Below is a conversation between a user and an assistant that describes a task, that provides context for the user's requirement. Write a response that appropriately completes the request.

### Input:
{}[/INST]

### Response:
{}"""

### INSTRUCTION ENGLISH
prompt_template_inst_english = """[INST] Below is an input that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Input:
{}[/INST]

### Response:
{}"""

"""--------------------------------------------------------------------
Utilizing Seamless as the English Intermediary Model
--------------------------------------------------------------------"""

""" 
translator = Translator("seamlessM4T_medium", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:0"),dtype=torch.float16)