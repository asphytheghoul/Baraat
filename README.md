


  

# **Project**  **'Baarat'**  **Proposal**  **and**  **Roadmap**

![Project Baarat](https://github.com/asphytheghoul/Baarat/assets/52605103/8c1ba4c4-03e6-4067-9a8e-fb65b7d8a2e0)


Project Baarat is an open-source initiative to leverage the power of
LLMs on Indic-NLP tasks. We aim to build Continually pre-trained, Task
Specific Language Models in a Mixture of Experts (MoE) setup through
domain adaptive pre-training. We plan on making a **multilingual**
**and**  **cross-lingual** LLM that is :

  

> 1\) Pre-trained on a large text corpus containing various sources of
> knowledge including crawled wikipedia articles, textbooks, news,
> social media sites, magazines etc.

>

> 2\) Is continually pre-trained on different downstream tasks. We first
> train a 7B LLaMa-2 model on an unsupervised text corpus in the target
> language and save it as a base model. We have considered the following
> tasks as downstream tasks that will be incorporated in the fine-tuning
> process:

>

> ● Machine Translation 
> ● Text Summarization 
> ● Question Answering 
> ● Instruct Fine-Tuning

>

> (this list is subject to change and a few tasks may be added over time).

>

> 3\) Creates task-specific \'experts\' - adapters for each task across
> languages - and runs them together in a mixture of experts setup using
> switch transformers for routing.

  

###  **Bird's**  **Eye**  **View**  **of**  **the**  **Project**

  

Overview:

  

> ● Set up extended vocabulary tokenizers in target languages (Kannada,
> Hindi) to reduce the number of tokens required compared to the base
> LLaMa-2 tokenizer.

>

> ● Prepare pre-training data from subsets of cleaned IITB corpus plus
> external sources like textbooks and news.

>

> ● Pre-train base LLaMa-2 7B model on this corpus to create a model
> with world knowledge. Save this as Baarat-hi-base-0.1.

>

> ● Fine-tune this base model on supervised data for machine translation
> between English, Hindi, Romanized Hindi.

>

> ● Merge adapters for each task across languages into single task-specific experts to enable cross-lingual support.

>

> ● Use switch transformers to route inputs to correct experts and adapters.

  

#### **Detailed**  **Description**  **of**  **the**  **Project**

  

#### **1)**  **Setting**  **up**  **Tokenizers**

>

> ● One of the main features in this project is an extended vocabulary
> llama-2 tokenizer for the languages under consideration for the
> purpose of the experiment. The languages that are considered are
> Kannada and Hindi. Extending the LLaMa-2 tokenizer was proposed with
> just one simple objective in mind - "To reduce the number of tokens
> required to tokenize long sequences of text from other languages". We
> observed a large number of unknown tokens and splitting up of text
> from a language (Hindi) in ways that were inefficient, the tokenizer
> always looked towards splitting up words into smaller pieces, this process might
> prove effective with English and English native scripts, however, for
> scripts like Hindi and Kannada, this technique is not the best as it
> leads to large number of tokens required to process before producing
> an output which **highly**  **affects**  **the**  **inference**  **speed** of the LLaMa 2 model.

>

> ● We experimented with various methods to extend the vocabulary of the
> base llama-2 tokenizer, the main challenges we faced was to ensure the
> existing merges file doesn't get affected and the tokenizer should not
> degrade in performance with the base model, and on the other hand it
> should provide an improvement **mainly**  **for**  **improving** **inference**  **speed** . 
> We experimented a lot and evaluated the tokenizer's performance by tokenizing sample sentences in both English and Hindi and realised the existing tokens were getting affected when we modified the merges.txt file.

> ● What needed to be done then? Our sole motivation behind extending
> the vocabulary of the tokenizer was to reduce the tokens required to
> process a piece of text, we didn't need to change the way words are
> merged, doing this empirically gave us worse results while trying to
> fine-tuning the base model with the tokenizer (with modified
> vocabularies and merges).

> ● So we proceeded with the following implementation, which notably
> gave us the best results and did not affect the performance of the
> base model on common tasks such as text generation or question
> answering:

> 1\) We trained a **BPE (Byte Pair Encoding) Sentencepiece** tokenizer on
> both languages separately, creating two tokenizer (.model) files. We
> processed them using the sentencepiece_extractor.py files huggingface
> has provided and obtained the newly learnt vocabularies along with
> their token IDs.

>

> 2\) To prevent any conflicts with the existing tokenizer's vocabulary,
> we appended the language's vocabulary to the tokenizer's vocabulary.
> So the token IDs for the new tokens started from 32000 (the base llama
> 2 tokenizer consists of 31999 tokens). The new tokenizers had an added
> vocabulary of 16000 each leading to 48000 tokens in their
> vocabularies.

>

> ● We evaluated the tokenizer on sequences of varying length and
> noticed an average **reduction of 60%** in the number of tokens produced
> when tokenizing the sequences compared to the base LLaMa-2 tokenizer.

  

#### **2)**  **Preparing**  **the**  **data**

  

> ● We use a subsample of the cleaned and preprocessed dataset from IITB
> for the unsupervised pre-training step. We supplement this subsample
> with external sources of knowledge such as textbooks, magazines and
> news articles. The final pre-training corpus consists of around 30
> million rows of sentences in hindi, costing around 9 GB in storage.


> ● The data for each task was heavily referenced based on the AI4Bharat
> Samantar and INDIC NLP datasets, augmented with external sources of
> data.

  

#### **3)**  **Detailed**  **breakdown**  **of**  **Task**  **1**  **:**

**Pre-Training**


The motivation for this task was to make a base model that has worldly
knowledge of events, science and context in hindi. This task involves
training the base LLaMa-2 7B model on the pre-training corpus for text
generation given an input prompt in the native language (next-word
prediction). Given an input prompt such as a few words in hindi, the
model understands the context behind the prompt and generates
semantically appropriate text following the prompt. The dataset is
around 9 GB in size but we are still exploring different ways of making
a good subsample with a balanced representation of knowledge sources to
make a smaller dataset.

  

We pre-train the model using the Hindi extended tokenizer and save the
model's adapters and merge them to form a 16-bit vLLM, this is **the**
**new**  **base**  **model**  **which**  **we**  **will**  **use**  **for** **fine-tuning**  **on**  **each**  **task.**  **We**  **call**  **this**
**Baarat-\<lan\>-base-0.1**  **where**  **\<lan\>**  **stands**  **for**
**the**  **language**  **code**  **(hi,ka).**

  

**4)**  **Detailed**  **breakdown**  **of**  **Task**  **2**  **:**

**Machine**  **Translation**

  

Machine Translation is one of the tasks we fine-tune the Baarat-hi-base-0.1 using the hindi tokenizer on a supervised dataset consisting of a translation corpus from AI4Bharat and augmented it with a few examples containing idioms with their examples in the sentences.

We prepared the corpus and pushed it to huggingface, the corpus consists
of around 650000 rows of supervised training samples for translation. It
consists of the following translation options:

>  - English to Hindi 
>  - Hindi to English
>  - Romanized Hindi to Hindi 
>  - Hindi to Romanized Hindi

  

The corpus is around 500 MB in size. A similar corpus is being prepared
for Kannada. We trained a FastText model to perform text classification
to understand which language the input text belongs to.

  

**5)**  **Proposed**  **Solution**  **with**  **a**  **stepwise**  **diagram**

<img width="683" alt="Stepwise Diagram" src="https://github.com/asphytheghoul/Baarat/assets/52605103/c106c66c-894c-425e-970f-fae481746033">


Step 1 : Train new BPE Sentencepiece tokenizers (same architecture as
LLaMa's tokenizer) for different languages on a small text corpus taken
from a sample of the samantar dataset with a vocabulary of 16000 tokens.

Extended the vocabulary of the base tokenizer with the vocabulary of the
new tokenizers creating two tokenizers (hindi and kannada for now).

  

Step 2 : Load the base LLaMa-2 model and prepare it for pre-training on
the unsupervised pre-training corpus mentioned above for each language
separately. Train the base model with the respective tokenizer as per
the chosen language for one epoch and save the model, merge the adapters
back to form a 16-bit vLLM. This is the new base model and will be used
for further downstream tasks.

  

Step 3 : Load an instance of the new base model and prepare the model
for supervised fine tuning for each task separately. Train the model for
5000 steps and save the adapters for each language of the same task.

  

Step 4 : With the trained adapters for all tasks for both Hindi and
Kannada, we merge the adapters of the same task from different languages
into one expert. We propose a task-specific expert as opposed to a
language specific expert. This could enable cross-lingual support with
minimum added overheads during the routing stage.

  

Step 5 : Implement the Switch Transformers' Routing algorithm to route
inputs appropriately to the correct expert and in the expert chosen,
route the input to the appropriate adapter and its tokenizer to process
the query. The response is then returned to the user.

  

Team Baarat,\
  Akash Kamalesh and Anirudh Lakhotia, PES University
