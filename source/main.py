from utilities import *
import re
from transformers import TextStreamer
import gradio as gr

def main():
    """--------------------------------------------------------------------
    Load models and tokenizers
    --------------------------------------------------------------------"""
    hindi_model = AutoModelForCausalLM.from_pretrained(
                "AsphyXIA/baarat-MOE-Hindi-v0.3", device_map='auto', torch_dtype=torch.bfloat16
    )
    hindi_tokenizer = AutoTokenizer.from_pretrained("AsphyXIA/baarat-MTH")

    kannada_model = AutoModelForCausalLM.from_pretrained(
                "AsphyXIA/baarat-MOE-Kannada-v0.1", device_map="auto", torch_dtype=torch.bfloat16
    )
    kannada_tokenizer = AutoTokenizer.from_pretrained("AsphyXIA/baarat-MTK")

    "NOTE: The Kannada mixture of experts is still being trained and this is a very basic version for testing functionality. It is suggested to NOT USE the Kannada MOE for implementation purposes until this comment is removed!"

    eng_model,eng_tokenizer = FastLanguageModel.from_pretrained(
        model_name = "meta-llama/Llama-2-7b-hf", 
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = True,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    """--------------------------------------------------------------------
    Prepare and pre-process text for the models
    --------------------------------------------------------------------"""

    def extract_context_and_question(text):
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

    """
    Example usage
    text = "question: what is ram's name? where is he from? what school is he from? based on the context: ram is a small boy from india."
    context, questions = extract_context_and_question(text)
    print("Context:", context)
    print("Questions:", questions)

    OUTPUT : 
    Context: ram is a small boy from india.
    Questions: what is ram's name? where is he from? what school is he from?
    """

    """--------------------------------------------------------------------
    Chain of functions to format the prompt based on the task.
    --------------------------------------------------------------------"""

    def predict_language(text,language):
        input_language = dict_map[lang_decoder.predict(text)[0][0]] 
        target_language = language
        if target_language in ["Kannada","Hindi","English"]:
            return {"input_language":input_language,"target_language":target_language}
        else:
            raise Exception('Please choose a valid target language from the following list: ["Kannada","Hindi","English"]')        

    # Function to process input based on language
    def route_input(text,languages_dict):
        src_language = languages_dict["input_language"]
        tgt_language = languages_dict["target_language"]
        routed_model = ""
        if (src_language == "English" and tgt_language !="English") or (tgt_language=="English" and src_language!="English"):
            routed_model = src_language if src_language!="English" else tgt_language
        elif src_language and tgt_language == "English":
            routed_model = "English"
        # cross lingual support implementation v1
        elif src_language !="English" and tgt_language !="English":
            lang_map = {"Kannada":"kan","English":"eng","Hindi":"hin"}
            translator = Translator("seamlessM4T_medium", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:0"),dtype=torch.float16)
            text = translator.predict(text, "t2tt", "eng" ,src_lang=lang_map[src_language])
            text = str(text[0][0])
            routed_model = tgt_language
        else:
            raise ValueError("An Unexpected Error has occured. Please try again.")
        return (routed_model,text)

    # Function that routes to the language expert based on the input
    def model_route(routed_model):
        if routed_model == "English":
            model = eng_model
            tokenizer = eng_tokenizer
        elif routed_model == "Hindi":
            model = hindi_model
            tokenizer = hindi_tokenizer
        elif routed_model == "Kannada":
            model = "kannada_model"
            tokenizer = "kannada_tokenizer"
        return (model,tokenizer)

    # Function which formats the input prompt appropriately
    def select_task(input_text,languages_dict,model,tokenizer):
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
                prompt_selected = prompt_template_inst_kannada
                prompt_selected = prompt_selected.format(languages_dict["input_language"],languages_dict["target_language"],input_text,"")
                return prompt_selected
            else:
                prompt_selected = prompt_template_inst_english
                prompt_selected = prompt_selected.format(input_text,"")
                return prompt_selected
        if not prompt_selected:
            raise Exception("An unexpected error occurred.")

        return prompt_selected

    def process_input(input_text,languages_dict,prompt_selected,model,tokenizer):
        inputs = tokenizer(prompt_selected,return_tensors = "pt").to("cuda")
        text_streamer = TextStreamer(tokenizer)
        _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 200,temperature=0.5,do_sample=True)

    """
    Example usage:
    input_text = "translate: ಆಕಾಶವು ಈಗ ನೀಲಿಯಾಗಿದೆ."
    target_language = "Hindi"
    languages_dict = predict_language(input_text,target_language)
    print(languages_dict)
    routed_model,text = route_input(input_text,languages_dict)
    print(routed_model,text)
    model,tokenizer = model_route(routed_model)
    prompt_select = select_task(text,languages_dict,model,tokenizer)
    print(prompt_select)

    OUTPUT:
    {'input_language': 'Kannada', 'target_language': 'Hindi'}
    Using the cached checkpoint of seamlessM4T_medium. Set `force` to `True` to download again.
    Using the cached tokenizer of seamlessM4T_medium. Set `force` to `True` to download again.
    Using the cached checkpoint of vocoder_36langs. Set `force` to `True` to download again.
    Hindi Translate: The sky is now blue.
    [INST] Below is a conversation between a user and an assistant that describes a task, that provides context for the user's requirement. Write a response that appropriately completes the request.

    ### Input:
    Translate: The sky is now blue.[/INST]

    ### Response:
    """

    def process_text(text, target_language, temperature,max_tokens):
        languages_dict = predict_language(text, target_language)
        routed_model, text = route_input(text, languages_dict)
        model, tokenizer = model_route(routed_model)
        prompt_selected = select_task(text, languages_dict, model, tokenizer)
        
        inputs = tokenizer(prompt_selected, return_tensors="pt").to("cuda")
        text_streamer = TextStreamer(tokenizer)
        output = model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_tokens, temperature=temperature, do_sample=True)
        
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return result

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

if __name__ == "__main__":
    main()

    
