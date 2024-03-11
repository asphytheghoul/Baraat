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



if __name__ == "__main__":
    main()

    
