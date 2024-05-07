# Project Baraat: Empowering Regional Languages in India 🇮🇳 with AI

## View the project on huggingface over [here](https://huggingface.co/projectbaraat)

<div align="center">
   
   # Project Baraat 🎉
   
   ![baraat](https://github.com/asphytheghoul/Baraat/assets/91832216/f3438f2e-0c52-46b8-ae03-60764387d1f6)

</div>

Project Baraat is an open-source initiative to leverage the power of
LLMs on Indic-NLP tasks. We aim to build Continually pre-trained, Task
Specific Language Models in a Mixture of Experts (MoE) setup. We plan on making a **multilingual**
**and**  **cross-lingual** LLM that is :

  
- 1\) Pre-trained on a large text corpus containing various sources of
knowledge including crawled wikipedia articles, textbooks, news,
social media sites, magazines etc.

- 2\) Fine-tuned on different downstream tasks. We first
train a 7B LLaMa-2 model on a text corpus in the target
language and save it as a base model. We have considered the following
tasks as downstream tasks that will be incorporated in the fine-tuning
process:

1. Machine Translation 
2. Mathematical and Logical Reasoning
3. Question Answering 
4. Instruct Fine-Tuning


> [!NOTE]
> This list is subject to change and a few tasks may be added over time.

| Model Tutorial | Notebook Link |
|-----------------|---------------|
| Baraat-hindi-experts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uLxQtMaDTJ_JLkVDbVrw2eeO6hkHE1XS?usp=sharing) |

  
## About Project Baraat 📖

Project Baraat is dedicated to making indigenous (regional) languages more accessible. With a focus on the rich linguistic diversity of India. This project aims to break language barriers and promote inclusivity through technology.
<br/>

## Roadmap 🎯
![image](https://github.com/asphytheghoul/Baraat/assets/91832216/6491ee06-382e-4d3f-87a8-47bdb3e4a3f4)
![image](https://github.com/asphytheghoul/Baraat/assets/91832216/fd33771e-6b4b-43a3-adc8-3c30820db6ec)


### Pre-trained Language Models and Datasets

| Model Name | Description | Dataset Link |
|------------|--------------|--------------|
| _**Baraat-hindi-pretrained**_ | Base model pre-trained on a diverse collection of datasets: <br><br> &#8226; [IndicCorp](https://ai4bharat.iitm.ac.in/indiccorp/): A multilingual corpus covering 9 major Indic languages for various NLP tasks. <br> &#8226; [Hindi Wikipedia Articles (172K)](https://www.kaggle.com/datasets/disisbig/hindi-wikipedia-articles-172k): A dataset containing 172,000 Hindi Wikipedia articles. <br> &#8226; [Hindi Corpus from Leipzig University](https://wortschatz.uni-leipzig.de/en/download/Hindi): A Hindi corpus provided by the University of Leipzig. <br> &#8226; [Animals: A Visual Encyclopedia](https://archive.org/details/animalsavisualencyclopedia/page/n133/mode/2up): An encyclopedia of general animal sentences. <br> &#8226; Augmented rows using Bing AI to include worldly knowledge such as fruits, vegetables, animals. | [Link](https://huggingface.co/datasets/projectbaraat/hindi-pretraining-data-v0.1) |
| _**Baraat-kannada-pretrained**_ | Base model pre-trained on a diverse collection of datasets: <br><br> &#8226; [IndicCorp](https://ai4bharat.iitm.ac.in/indiccorp/): A multilingual corpus covering 9 major Indic languages for various NLP tasks. <br> &#8226; [Kannada Corpus from Leipzig University](https://wortschatz.uni-leipzig.de/en/download/Kannada): A Kannada corpus provided by the University of Leipzig. | [Link](https://huggingface.co/datasets/projectbaraat/kannada-pretraining-data-v0.1) |
### Key Features ✨

- **Tokenizers for Indian Languages**: Robust tokenization tools tailored for the unique structures of regional Indian languages.
- **Fine-tuned Language Models**: Leveraging the power of Large Language Models (LLMs) fine-tuned for Indian languages to understand and generate text with high accuracy.
- **Open Source Collaboration**: We believe in the collective power of the community to drive innovation and inclusivity. 🤝
- **High Quality Datasets**: Take a look at our suite of cleaned datasets ready for your own downstream training purposes.
<br/>

## Architecture ✏️

![Architecture](https://github.com/asphytheghoul/Baraat/assets/91832216/a0cbd07f-3c2a-4569-9674-993b57713a7e)


## Our Vision 🌟

To promote the spirit of building accessible models in native languages, fostering a world where technology speaks everyone's language. 🗣️
<br/>
<br/>

## Roadmap 🛣️

- ✅ Prepare and setup dataset
- ✅ Prepare and setup tokenizers
- ✅ Start pre-training
- ✅ Fine-tune models
- ✅ Implement gating mechanism
- ✅ Implement MoE
- ✅ Simple Demo


Foundational model: LLaMa-2 7B
<br/>

## Small Demo of the project

P.S. The project is still in its early stages and this is a Proof of Concept implementation for **Hindi**. 

https://github.com/asphytheghoul/Baraat/assets/91832216/74aae2d7-818b-40eb-af43-ad955bbf6d45

- We can see here that the model is sensitive to the prompts that are being passed to it and this is a feature prevelant in a wide variety of LLMs today. We aim to train our suite of models for a longer period of time with evaluation steps.
- The project is being worked on actively and is currently undergoing an update. All utility files are provided in the source directory.

## Future Scope 🔜

- ### Extending Support for Images and Audio

In the future, we aim to expand Project Baraat's capabilities beyond text to include support for images and audio, enabling multimodal learning techniques.

- ### Pipeline for Automated Dataset Cleaning

We plan to develop a pipeline for dataset cleaning, leveraging small models like [stabilityai/stablelm-zephyr-3b](https://huggingface.co/stabilityai/stablelm-zephyr-3b) or [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) for automated data cleaning processes.

- ### Enhanced Reasoning Ability in Fine-Tuning

We intend to introduce an additional step in fine-tuning to enhance the model's reasoning ability, integrating techniques for logical reasoning and inferencin using datasets like [meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) or [microsoft/orca-math-word-problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k).
We plan to release translated versions of the datasets to facilitate research in mathematical reasoning and question answering across diverse linguistic communities.



## Contribute to Project Baraat 🛠️

We welcome open-source contributions! Whether you're a coder, a linguist, or just someone passionate about language accessibility, there's a place for you in Project Baraat. Here's how you can get involved:

1. **Star and Fork**: Give us a star ⭐ on GitHub and fork the repository to start contributing.
2. **Issue Tracker**: Report bugs or suggest new features by creating an issue.
3. **Pull Requests**: Submit your pull requests with new features, bug fixes, or documentation enhancements.

Check out our [CONTRIBUTING.md](./CONTRIBUTING.md) for more detailed guidelines.
<br/>

## Additional Contributions:

- ### Sentence Chunking for Enhanced Pretraining:

We partition sentences from datasets into chunks of predetermined maximum word count. This approach allows for the creation of extended sentences, thereby significantly augmenting the efficacy of the continual pretraining process.
This can be applied to any dataset to combine sentences and produce a new dataset with more content per row.

- ### Token Counting for Diverse Tokenizers and Datasets:

A token counting mechanism has been integrated, capable of quantifying the number of tokens within any given dataset for any given tokenizer. This feature serves as a fundamental tool for analyzing token distributions and comprehending vocabulary dimensions across datasets.
We built this by modifying Sayak Paul's [count-tokens-hf-datasets](https://github.com/sayakpaul/count-tokens-hf-datasets/) project. We no longer require Google Cloud as a component to count tokens, and the entire process can be performed locally.

- ### Token Distribution Visualization and Binning:

We also visualize token distributions within individual sentences of datasets. Additionally, a binning process has been implemented to enhance the interpretability of token distribution patterns. These enhancements provide valuable insights into the structural characteristics of textual data, benefiting both researchers and practitioners.

## License 📄

Project Baraat is released under the [MIT License](./LICENSE).
<br/>

## Show Your Support 🌈

If you like Project Baraat, please consider starring the repository and sharing it with your network!
<br/>

---

Made with ❤️ by Team Baraat,\
  [Akash Kamalesh](https://github.com/asphytheghoul) , [Anirudh Lakhotia](https://github.com/anirudhlakhotia/) and [Tanistha Hota](https://github.com/hota15), PES University, Bengaluru.



