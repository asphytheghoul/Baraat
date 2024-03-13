# Project Baraat

![image](https://github.com/asphytheghoul/Baarat/assets/91832216/08ba0790-c916-4071-a184-aff4bb67b52e)

## Mixture of Experts Details
A major part of the project was to create powerful open-source language specific models that are trained a large suite of datasets and tasks. We have built Continually pre-trained, Task Specific Language Models in a Mixture of Experts (MoE) setup utilizing MergeKit as the underlying merging platform. The models are trained on a large text corpus containing various sources of knowledge including crawled wikipedia articles, textbooks, news, social media sites, magazines etc. All language experts are **multi-lingual** and **cross-lingual**.

### Hindi Expert
We identified pressing tasks out of a variety of domains. Namely, 
- **Translation**
- **Instruct Tuning** 
- **Question Answering** 
- **Logical Reasoning**

### Steps Followed While Training Invidual Experts
1) The Base Model is trained in an unsupervised fashion incorporating Causal Language Modeling for next word prediction on a large text corpus containing various sources of knowledge including crawled Wikipedia articles, textbooks, news, social media sites, magazines etc. on a 7B LLaMa-2 model, utilizing the unsloth AI framework with Low Rank Adaptation Adapters. We then proceed to merge the trained adapters into a 16-bit vLLM. This will serve as a base model for the language.

2) Once we have the base model, we proceed to fine-tune the model on a variety of tasks using Supervised Fine-Tuning on the tasks we had mentioned above for about 30000 steps with a training batch size of 4 and gradient accumulation steps of 2. We use the AdamW optimizer with a learning rate of 2e-4 and a weight decay of 0.01. The models are fine-tuned using Low Rank Adaptation Adapters with Unsloth AI.

3) We then proceed to merge the fine-tuned adapters into 16-bit vLLMs. This marks the stage where we have our individual task specific experts ready for integration into the MoE.

**All operations were run on a single compute node with 220GB of RAM and an NVIDIA A100 GPU with 80GB of memory.**

