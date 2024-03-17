from .utilities import *
import re
from transformers import TextStreamer
import gradio as gr
from unsloth import FastLanguageModel
import torch
from transformers import AutoTokenizer
max_seq_length = 2048 
dtype = None 
load_in_4bit = False 
import fasttext
import warnings

warnings.filterwarnings('ignore')
fasttext.FastText.eprint = lambda x: None


def load_individual_models(path,load_in_4bit=False):
    """
    load an individual expert. choose one expert from huggingface.co/projectbaraat
    this function utilizes unsloth internally.
    model,tokenizer = load_individual_models(path,load_in_4bit=False)
    """
    from unsloth import FastLanguageModel
    model, tokenizer= FastLanguageModel.from_pretrained(
        model_name = path, # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    return model,tokenizer

def load_models():
    """
    Load models and tokenizers.
    Usage : 
    hindi_model, hindi_tokenizer, eng_model, eng_tokenizer = load_models()
    """
    hindi_model = AutoModelForCausalLM.from_pretrained(
            "AsphyXIA/baarat-MOE-Hindi-v0.3", device_map='auto', torch_dtype=torch.bfloat16
    )
    hindi_tokenizer = AutoTokenizer.from_pretrained("AsphyXIA/baarat-MTH")

    # kannada_model = AutoModelForCausalLM.from_pretrained(
    #         "AsphyXIA/baarat-MOE-Kannada-v0.1", device_map="auto", torch_dtype=torch.bfloat16
    # )
    # kannada_tokenizer = AutoTokenizer.from_pretrained("AsphyXIA/baarat-MTK")

    eng_model,eng_tokenizer = FastLanguageModel.from_pretrained(
        model_name = "meta-llama/Llama-2-7b-hf", 
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = True,
    )
    return hindi_model, hindi_tokenizer, eng_model, eng_tokenizer

def extract_context_and_question(text):
    """
    Extract context and question from the text.
    Example usage:
    text = "question: what is ram's name? where is he from? what school is he from? based on the context: ram is a small boy from india."
    context, questions = extract_context_and_question(text)
    print("Context:", context)
    print("Questions:", questions)
    
    OUTPUT : 
    Context: ram is a small boy from india.
    Questions: what is ram's name? where is he from? what school is he from?
    """
    # Check if "context:" and "question:" are present in the text
    context_present = re.search(r'context:', text)
    question_present = re.search(r'question:', text)
    
    context = ""
    questions = []
    
    if context_present and question_present:
        # Both "context:" and "question:" are present
        context_match = re.search(r'context:(.*?)(?=question:|$)', text, re.DOTALL)
        question_match = re.search(r'question:(.*?)(?=context:|$)', text, re.DOTALL)
        
        if context_match:
            context = context_match.group(1).strip()
        if question_match:
            # Split the questions at '? ' followed by 'where' or 'what'
            questions = re.split(r'\? (?=where|what)', question_match.group(1).strip())
            # Remove any trailing 'based on the' from the last question
            questions = [re.sub(r'\s+based on the$', '', q) for q in questions]
    elif context_present:
        # Only "context:" is present
        context = re.search(r'context:(.*)', text, re.DOTALL).group(1).strip()
    elif question_present:
        # Only "question:" is present
        questions_text = re.search(r'question:(.*)', text, re.DOTALL).group(1).strip()
        questions = re.split(r'\? (?=where|what)', questions_text)
        questions = [re.sub(r'\s+based on the$', '', q) for q in questions]
    else:
        # Neither "context:" nor "question:" is present
        return "Neither context nor question found in the text."
    questions = [item+"?" if "?" not in item else item for item in questions]
    questions = " ".join(questions)
    return context, questions

def predict_language(text,language):
    """
    Predict the source language of the text, pass the target language of the text. returns a dictionary with the source language and target language as : 
    {"input_language":input_language,"target_language":target_language}
    source_language = predict_language(text,language)
    """
    lang_decoder = get_lang_decoder()
    dict_map = get_dict_map()
    input_language = dict_map[lang_decoder.predict(text)[0][0]] 
    target_language = language
    if target_language in ["Kannada","Hindi","English"]:
        return {"input_language":input_language,"target_language":target_language}
    else:
        raise Exception('Please choose a valid target language from the following list: ["Kannada","Hindi","English"]')        

def route_input(text,languages_dict):
    """
    Route the input based on the language.
    model,text = route_input(text,languages_dict)
    """
    src_language = languages_dict["input_language"]
    tgt_language = languages_dict["target_language"]
    routed_model = ""
    if (src_language == "English" and tgt_language !="English") or (tgt_language=="English" and src_language!="English"):
        routed_model = src_language if src_language!="English" else tgt_language
    elif src_language and tgt_language == "English":
        routed_model = "English"
    elif src_language !="English" and tgt_language !="English":
        lang_map = {"Kannada":"kan","English":"eng","Hindi":"hin"}
        translator = Translator("seamlessM4T_medium", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:0"),dtype=torch.float16)
        text = translator.predict(text, "t2tt", "eng" ,src_lang=lang_map[src_language])
        text = str(text[0][0])
        routed_model = tgt_language
    else:
        raise ValueError("An Unexpected Error has occured. Please try again.")
    return (routed_model,text)

def model_route(routed_model):
    """
    Route to the language expert based on the input.
    routed_model = model_route(routed_model)
    """
    if routed_model == "English":
        model = eng_model
        tokenizer = eng_tokenizer
    elif routed_model == "Hindi":
        model = hindi_model
        tokenizer = hindi_tokenizer
    elif routed_model == "Kannada":
        # model = kannada_model
        # tokenizer = kannada_tokenizer
        raise ValueError("The kannada model is not supported right now and is under development.")
    return (model,tokenizer)

def select_task(input_text,languages_dict,model,tokenizer):
    """
    Select the task based on the input text.
    selected_prompt = select_task(input_text,languages_dict,model,tokenizer)
    """
    if "translate" in input_text:
        prompt_selected = translation_prompt
        prompt_selected = prompt_selected.format(languages_dict["input_language"],languages_dict["target_language"],input_text,"")
        return prompt_selected
    if "question" in input_text:
        prompt_selected = q_a_prompt
        context,question = extract_context_and_question(input_text)
        if question =="":
            raise ValueError("No question was present in your prompt. Please try again")
        prompt_selected = q_a_prompt.format(context,question,"")
        return prompt_selected
    else:
        if languages_dict["target_language"]=="Hindi":
            prompt_selected = prompt_template_inst_hindi
            prompt_selected = prompt_selected.format(input_text,"")
            return prompt_selected
            
        elif languages_dict["target_language"]=="Kannada":
            raise ValueError("The kannada model is not supported right now and is under development.")
            # prompt_selected = prompt_template_inst_kannada
            # prompt_selected = prompt_selected.format(languages_dict["input_language"],languages_dict["target_language"],input_text,"")
            # return prompt_selected
        else:
            prompt_selected = prompt_template_inst_english
            prompt_selected = prompt_selected.format(input_text,"")
            return prompt_selected
    if not prompt_selected:
        raise Exception("An unexpected error occurred.")

    return prompt_selected

def process_text(text, target_language,prompt,model,tokenizer, temperature=1,max_tokens=200):
    """
    Process the text for translation.
    result = process_text(text, target_language,model,tokenizer, temperature=1,max_tokens=200)
    """
    languages_dict = predict_language(text, target_language)
    routed_model, text = route_input(text, languages_dict)
    # model, tokenizer = model_route(routed_model)
    # prompt_selected = select_task(text, languages_dict, model, tokenizer)
    src_lan = languages_dict["input_language"]
    if target_language not in ["Hindi","English"]:
        raise NotImplementedError('Language has to be one of "Hindi" or "English"')
    prompt_selected = prompt.format(src_lan,target_language,text,"")
    inputs = tokenizer(prompt_selected, return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)
    model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
    
def launch_gradio():
    """
    Launch the Gradio Interface
    """
    iface = gr.Interface(
        fn=process_text, # The function to call
        inputs=[
            gr.Textbox(lines=2, placeholder="Enter text here..."),
            gr.Dropdown(choices=["English", "Hindi", "Kannada"], label="Target Language"),
            gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.5, label="Temperature"),
            gr.Slider(minimum=100,maximum=512,step=100,value=200,label="Maximum Tokens")
        ],
        outputs="text"
    )

    iface.launch(share=True)
