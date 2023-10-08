# Awesome-Refreshing-LLMs <!-- omit from toc -->

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
![](https://img.shields.io/badge/PRs-Welcome-red)
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/hyintell/awesome-refreshing-llms/main?logo=github&color=blue)

Although **large language models (LLMs)** are impressive in solving various tasks, they can quickly be outdated after deployment. Maintaining their up-to-date status is a pressing concern in the current era. How can we refresh LLMs to align with the ever-changing world knowledge ***without expensive retraining from scratch***?


<p align=center>
    <img src="./images/llm_align_world_cropped.jpg" width="60%" height="60%" alt="llm_align_world_example"/>
    <br>
    <em>An LLM after training is static and can be quickly outdated. For example, <a href="https://openai.com/blog/chatgpt" target="_blank">ChatGPT</a> has a knowledge <br> cutoff date of September 2021. Without <a href="https://openai.com/blog/chatgpt-plugins" target="_blank">web browsing</a>, it does not know the latest information ever since.</em>
</p>


## 📢 News
- **[2023-10] Our survey paper: *"How Do Large Language Models Capture the Ever-changing World Knowledge? A Review of Recent Advances"* has been accepted by [EMNLP 2023](https://2023.emnlp.org/)! We will release the camera-ready version soon.**
- **[2023-10] We create this repository to maintain a paper list on *refreshing LLMs without retraining*.**

---

## 🔍 Table of Contents

- [📢 News](#-news)
- [🔍 Table of Contents](#-table-of-contents)
- [📃 Papers](#-papers)
  - [Methods Overview](#methods-overview)
  - [Knowledge Editing](#knowledge-editing)
    - [Meta-learning](#meta-learning)
    - [Hypernetwork Editor](#hypernetwork-editor)
    - [Locate and Edit](#locate-and-edit)
  - [Continual Learning](#continual-learning)
    - [Continual Pre-training](#continual-pre-training)
    - [Continual Knowledge Editing](#continual-knowledge-editing)
  - [Memory-enhanced](#memory-enhanced)
  - [Retrieval-enhanced](#retrieval-enhanced)
  - [Internet-enhanced](#internet-enhanced)
- [💻 Resources](#-resources)
  - [Related Survey](#related-survey)
  - [Tools](#tools)
- [🚩 Citation](#-citation)
- [🎉 Acknowledgement \& Contribution](#-acknowledgement--contribution)


## 📃 Papers

### Methods Overview

To refresh LLMs to align with the ever-changing world knowledge without retraining, we roughly categorize existing methods into ***Implicit*** and ***Explicit*** approaches.
***Implicit*** means the approaches seek to directly alter the knowledge stored in LLMs, such as parameters or weights, while ***Explicit*** means more often incorporating external resources to override internal knowledge, such as augmenting a search engine.

Please see our paper for more details.

<p align=center>
    <img src="./images/taxonomy.png" width="75%" height="75%" alt="methods taxonomy"/>
    <br>
    <em>Taxonomy of methods to align LLMs with the ever-changing world knowledge.</em>
</p>

<p align=center>
    <img src="./images/compare_of_methods_cropped.jpg" width="75%" height="75%" alt="methods overview"/>
    <br>
    <em>A high-level comparison of different approaches.</em>
</p>



### Knowledge Editing

> **Knowledge editing (KE)** is an arising and promising research area that aims to alter the parameters of some specific knowledge stored in pre-trained models so that the model can make new predictions on those revised instances while keeping other irrelevant knowledge unchanged. 
> We categorize existing methods into *meta-learning*, *hypernetwork*, and *locate-and-edit* -based methods.

#### Meta-learning

| Year | Venue | Paper                                                   | Link                                                                                                                                                                                                                                                         |
| :--- | :---- | :------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023 | Arxiv | RECKONING: Reasoning through Dynamic Knowledge Encoding | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.06349)                                                                                                                                 |
| 2020 | ICLR  | Editable Neural Networks                                | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://openreview.net/forum?id=HJedXaEtvS) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/editable-ICLR2020/editable) |

#### Hypernetwork Editor

| Year | Venue | Paper                                                                               | Link                                                                                                                                                                                                                                                                |
| :--- | :---- | :---------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2023 | Arxiv | Inspecting and Editing Knowledge Representations in Language Models                 | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2304.00740) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/evandez/REMEDI)                              |
| 2023 | EACL  | Methods for Measuring, Updating, and Visualizing Factual Beliefs in Language Models | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2023.eacl-main.199/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/peterbhase/SLAG-Belief-Updating) |
| 2022 | ICLR  | Fast Model Editing at Scale                                                         | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://openreview.net/forum?id=0DcZxeWfOPt) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/eric-mitchell/mend)               |
| 2021 | EMNLP | Editing Factual Knowledge in Language Models                                        | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2021.emnlp-main.522/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/nicola-decao/KnowledgeEditor)   |

#### Locate and Edit

| Year | Venue   | Paper                                        | Link                                                                                                                                                                                                                                                                                                                     |
| :--- | :------ | :------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023 | Arxiv   | Editing Commonsense Knowledge in GPT         | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.14956) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/anshitag/memit_csk)                                                                               |
| 2023 | ICLR    | Mass-Editing Memory in a Transformer         | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://openreview.net/forum?id=MkbcAHIYgyS) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/kmeng01/memit)                                                                         |
| 2022 | ACL     | Knowledge Neurons in Pretrained Transformers | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.acl-long.581/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/hunter-ddm/knowledge-neurons)                                                          |
| 2022 | NeurIPS | Fast Model Editing at Scale                  | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/kmeng01/rome) |


### Continual Learning

> **Continual learning (CL)** aims to enable a model to learn from a continuous data stream across time while reducing catastrophic forgetting of previously acquired knowledge. With CL, a deployed LLM has the potential to adapt to the changing world without costly re-training from scratch. Below papers employ CL for aligning language models with the current world knowledge, including *Continual Pre-training* and *Continual Knowledge Editing*.

#### Continual Pre-training

| Year | Venue   | Paper                                                                                           | Link                                                                                                                                                                                                                                                                                                                                                               |
| :--- | :------ | :---------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023 | Arxiv   | KILM: Knowledge Injection into Encoder-Decoder Language Models                                  | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2302.09170) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/alexa/kilm)                                                                                                                                 |
| 2023 | Arxiv   | Semiparametric Language Models Are Scalable Continual Learners                                  | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2303.01421)                                                                                                                                                                                                                                       |
| 2023 | Arxiv   | Meta-Learning Online Adaptation of Language Models                                              | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.15076)                                                                                                                                                                                                                                       |
| 2023 | ICLR    | Continual Pre-training of Language Models                                                       | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://openreview.net/forum?id=m_GDIItaI3o) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/UIC-Liu-Lab/ContinualLM)                                                                                                         |
| 2023 | ICML    | Lifelong Language Pretraining with Distribution-Specialized Experts                             | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.12281)                                                                                                                                                                                                                                       |
| 2022 | ACL     | ELLE: Efficient Lifelong Pre-training for Emerging Data                                         | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.findings-acl.220/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/thunlp/elle)                                                                                                                 |
| 2022 | EMNLP   | Fine-tuned Language Models are Continual Learners                                               | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.emnlp-main.410/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/ThomasScialom/T0_continual_learning)                                                                                           |
| 2022 | EMNLP   | Continual Training of Language Models for Few-Shot Learning                                     | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.emnlp-main.695/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/UIC-Liu-Lab/CPT)                                                                                                               |
| 2022 | EMNLP   | TemporalWiki: A Lifelong Benchmark for Training and Evaluating Ever-Evolving Language Models    | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.emnlp-main.418/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/joeljang/temporalwiki)                                                                                                         |
| 2022 | ICLR    | LoRA: Low-Rank Adaptation of Large Language Models                                              | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://openreview.net/forum?id=nZeVKeeFYf9) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/microsoft/LoRA)                                                                                                                  |
| 2022 | ICLR    | Towards Continual Knowledge Learning of Language Models                                         | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://openreview.net/forum?id=vfsRB5MImo9) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/joeljang/continual-knowledge-learning)                                                                                           |
| 2022 | NAACL   | DEMix Layers: Disentangling Domains for Modular Language Modeling                               | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.naacl-main.407/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/kernelmachine/demix)                                                                                                           |
| 2022 | NAACL   | Lifelong Pretraining: Continually Adapting Language Models to Emerging Corpora                  | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.naacl-main.351/)                                                                                                                                                                                                                          |
| 2022 | NeurIPS | Factuality Enhanced Language Models for Open-Ended Text Generation                              | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2206.04624) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/nayeon7lee/FactualityPrompt)                                                                                                                |
| 2022 | TACL    | Time-Aware Language Models as Temporal Knowledge Bases                                          | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00459/110012/Time-Aware-Language-Models-as-Temporal-Knowledge) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/google-research/language/tree/master/language/templama) |
| 2021 | ACL     | K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters                             | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2021.findings-acl.121/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/microsoft/K-Adapter)                                                                                                         |
| 2021 | EACL    | Analyzing the Forgetting Problem in Pretrain-Finetuning of Open-domain Dialogue Response Models | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2021.eacl-main.95/)                                                                                                                                                                                                                            |
| 2020 | EMNLP   | Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting              | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2020.emnlp-main.634/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/Sanyuan-Chen/RecAdam)                                                                                                          |

#### Continual Knowledge Editing

| Year | Venue | Paper                                                                     | Link                                                                                                                                                                                                                                                              |
| :--- | :---- | :------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023 | Arxiv | Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adapters | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2211.11031) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/thartvigsen/grace)                         |
| 2023 | ICLR  | Transformer-Patcher: One Mistake Worth One Neuron                         | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://openreview.net/forum?id=4oYUGeGBPm) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/ZeroYuHuang/Transformer-Patcher) |
| 2022 | ACL   | On Continual Model Refinement in Out-of-Distribution Data Streams         | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.acl-long.223/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/facebookresearch/CMR)           |
| 2022 | ACL   | Plug-and-Play Adaptation for Continuously-updated QA                      | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.findings-acl.37/)                                                                                                                        |

### Memory-enhanced

> Pairing a static LLM with a growing **non-parametric memory** enables it to capture information beyond its memorized knowledge during inference. The external memory can store a recent *corpus* or *feedback* that contains new information to guide the model generation.

| Year | Venue | Paper                                                                                                         | Link                                                                                                                                                                                                                                                        |
| :--- | :---- | :------------------------------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023 | Arxiv | Adaptation Approaches for Nearest Neighbor Language Models                                                    | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2211.07828)                                                                                                                                |
| 2023 | Arxiv | Semiparametric Language Models Are Scalable Continual Learners                                                | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2303.01421)                                                                                                                                |
| 2023 | Arxiv | MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions                                | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.14795) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/princeton-nlp/MQuAKE)                |
| 2022 | EMNLP | You can’t pick your neighbors, or can you? When and How to Rely on Retrieval in the kNN-LM                    | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.findings-emnlp.218/)                                                                                                               |
| 2022 | EMNLP | Nearest Neighbor Zero-Shot Inference                                                                          | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.emnlp-main.214/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/swj0419/kNN_prompt)     |
| 2022 | EMNLP | Memory-assisted prompt editing to improve GPT-3 after deployment                                              | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.emnlp-main.183/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/madaan/memprompt)       |
| 2022 | EMNLP | Towards Teachable Reasoning Systems: Using a Dynamic Memory of User Feedback for Continual System Improvement | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.emnlp-main.644/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://allenai.org/data/teachme)          |
| 2022 | ICML  | Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval                                           | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2201.12431) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/neulab/retomaton)                    |
| 2022 | ICML  | Memory-Based Model Editing at Scale                                                                           | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2206.06520) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/eric-mitchell/serac)                 |
| 2022 | NAACL | Learning to repair: Repairing model output errors after deployment using a dynamic memory of feedback         | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2022.findings-naacl.26/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/allenai/interscript) |
| 2021 | EMNLP | Efficient Nearest Neighbor Language Models                                                                    | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2021.emnlp-main.461/) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/jxhe/efficient-knnlm)   |
| 2021 | EMNLP | BeliefBank: Adding Memory to a Pre-Trained Language Model for a Systematic Notion of Belief                   | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://aclanthology.org/2021.emnlp-main.697/)                                                                                                                   |
| 2020 | ICLR  | Generalization through Memorization: Nearest Neighbor Language Models                                         | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://openreview.net/forum?id=HklBjCEKvH) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/urvashik/knnlm)            |

### Retrieval-enhanced

> Leveraging an off-the-shelf retriever and the in-context learning ability of LLMs, this line of work designs better retrieval strategies to incorporate world knowledge into a fixed LLM through prompting, which can be divided into *single-stage* and *multi-stage*.


<p align=center>
    <img src="./images/single_multi_stage_cropped.jpg" width="50%" height="50%" alt="single_and_multiple_stage_retrieval"/>
    <br>
    <em>Single-Stage (left) typically retrieves once, while Multi-Stage (right) involves multiple retrievals or revisions to solve complex questions</em>
</p>



| Year | Venue | Paper                                                                                                          | Link                                                                                                                                                                                                                                                             |
| :--- | :---- | :------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023 | ACL   | Augmentation-Adapted Retriever Improves Generalization of Language Models as Generic Plug-In                   | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.17331) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/OpenMatch/Augmentation-Adapted-Retriever) |
| 2023 | ACL   | When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories       | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2212.10511) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/AlexTMallen/adaptive-retrieval)           |
| 2023 | ACL   | Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions            | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2212.10509) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/stonybrooknlp/ircot)                      |
| 2023 | ACL   | RARR: Researching and Revising What Language Models Say, Using Language Models                                 | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2210.08726) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/anthonywchen/RARR)                        |
| 2023 | ACL   | MultiTool-CoT: GPT-3 Can Use Multiple External Tools with Chain of Thought Prompting                           | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.16896) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/InabaTatsuro/MultiTool-CoT)               |
| 2023 | Arxiv | Can We Edit Factual Knowledge by In-Context Learning?                                                          | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.12740) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/Zce1112zslx/IKE)                          |
| 2023 | Arxiv | REPLUG: Retrieval-Augmented Black-Box Language Models                                                          | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2301.12652)                                                                                                                                     |
| 2023 | Arxiv | Improving Language Models via Plug-and-Play Retrieval Feedback                                                 | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.14002)                                                                                                                                     |
| 2023 | Arxiv | Measuring and Narrowing the Compositionality Gap in Language Models                                            | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2210.03350) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/ofirpress/self-ask)                       |
| 2023 | Arxiv | ART: Automatic multi-step reasoning and tool-use for large language models                                     | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2303.09014) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/bhargaviparanjape/language-programmes/)   |
| 2023 | Arxiv | ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models                         | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.14323) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/RUCAIBOX/ChatCoT)                         |
| 2023 | Arxiv | Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2302.12813) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/pengbaolin/LLM-Augmenter)                 |
| 2023 | Arxiv | Question Answering as Programming for Solving Time-Sensitive Questions                                         | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.14221) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/microsoft/ContextualSP/tree/master/qaap)  |
| 2023 | Arxiv | Active Retrieval Augmented Generation                                                                          | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.06983) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/jzbjyb/FLARE)                             |
| 2023 | Arxiv | Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP                | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2212.14024) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/stanfordnlp/dspy)                         |
| 2023 | Arxiv | Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy                | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.15294)                                                                                                                                     |
| 2023 | Arxiv | Verify-and-Edit: A Knowledge-Enhanced Chain-of-Thought Framework                                               | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.03268)                                                                                                                                     |
| 2023 | Arxiv | CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing                                | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.11738) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/microsoft/ProphetNet/tree/master/CRITIC)  |
| 2023 | Arxiv | WikiChat: A Few-Shot LLM-Based Chatbot Grounded with Wikipedia                                                 | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.14292)                                                                                                                                     |
| 2023 | Arxiv | Query Rewriting for Retrieval-Augmented Large Language Models                                                  | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.14283)                                                                                                                                     |
| 2023 | ICLR  | Prompting GPT-3 To Be Reliable                                                                                 | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://openreview.net/forum?id=98p5x51L5af) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/NoviScl/GPT3-Reliability)      |
| 2023 | ICLR  | Decomposed Prompting: A Modular Approach for Solving Complex Tasks                                             | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://openreview.net/forum?id=_nGgzQjzaRy) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/allenai/DecomP)                |
| 2023 | ICLR  | ReAct: Synergizing Reasoning and Acting in Language Models                                                     | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://openreview.net/forum?id=WE_vluYUL-X) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/ysymyth/ReAct)                 |
| 2023 | TACL  | In-Context Retrieval-Augmented Language Models                                                                 | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2302.00083) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/AI21Labs/in-context-ralm)                 |
| 2022 | Arxiv | Rethinking with Retrieval: Faithful Large Language Model Inference                                             | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2301.00303) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/HornHehhf/RR)                             |

### Internet-enhanced

> A recent trend uses the whole web as the knowledge source and equips LLMs with the **Internet** to support real-time information seeking.

| Year | Venue | Paper                                                                                            | Link                                                                                                                                                                                                                                                                 |
| :--- | :---- | :----------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2023 | ACL   | Large Language Models are Built-in Autoregressive Search Engines                                 | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.09612) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/Ziems/llm-url)                                |
| 2023 | ACL   | RARR: Researching and Revising What Language Models Say, Using Language Models                   | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2210.08726) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/anthonywchen/RARR)                            |
| 2023 | Arxiv | Measuring and Narrowing the Compositionality Gap in Language Models                              | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2210.03350) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/ofirpress/self-ask)                           |
| 2023 | Arxiv | ART: Automatic multi-step reasoning and tool-use for large language models                       | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2303.09014) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/bhargaviparanjape/language-programmes/)       |
| 2023 | Arxiv | TaskMatrix.AI: Completing Tasks by Connecting Foundation Models with Millions of APIs            | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2303.16434) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/microsoft/TaskMatrix/tree/main/TaskMatrix.AI) |
| 2023 | Arxiv | MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action                                  | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2303.11381) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/microsoft/MM-REACT)                           |
| 2023 | Arxiv | Active Retrieval Augmented Generation                                                            | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.06983) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/jzbjyb/FLARE)                                 |
| 2023 | Arxiv | Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models                      | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2304.09842) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/lupantech/chameleon-llm)                      |
| 2023 | Arxiv | CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing                  | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.11738) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/microsoft/ProphetNet/tree/master/CRITIC)      |
| 2023 | Arxiv | Query Rewriting for Retrieval-Augmented Large Language Models                                    | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2305.14283)                                                                                                                                         |
| 2023 | ICLR  | ReAct: Synergizing Reasoning and Acting in Language Models                                       | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://openreview.net/forum?id=WE_vluYUL-X) [![Static Badge](https://img.shields.io/badge/code-black?logo=github)](https://github.com/ysymyth/ReAct)                     |
| 2022 | Arxiv | Internet-augmented language models through few-shot prompting for open-domain question answering | [![Static Badge](https://img.shields.io/badge/paper-%23B31B1B?logo=arxiv&labelColor=grey)](https://arxiv.org/abs/2203.05115)                                                                                                                                         |

## 💻 Resources

### Related Survey

- [Augmented Language Models: a Survey](https://arxiv.org/abs/2302.07842), 2023
- [The Life Cycle of Knowledge in Big Language Models: A Survey](https://arxiv.org/abs/2303.07616), 2023
- [Interactive Natural Language Processing](https://arxiv.org/abs/2305.13246), 2023
- [Editing Large Language Models: Problems, Methods, and Opportunities](https://arxiv.org/abs/2305.13172), 2023
- [Tool Learning with Foundation Models](https://arxiv.org/abs/2304.08354), 2023
- [Unifying Large Language Models and Knowledge Graphs: A Roadmap](https://arxiv.org/abs/2306.08302), 2023
- [A Review on Language Models as Knowledge Bases](https://arxiv.org/abs/2204.06031), 2022
- [A Survey of Knowledge-enhanced Text Generation](https://dl.acm.org/doi/10.1145/3512467), 2022
- [A Survey of Knowledge-Intensive NLP with Pre-Trained Language Models](https://arxiv.org/abs/2202.08772), 2022
- [A Survey on Knowledge-Enhanced Pre-trained Language Models](https://arxiv.org/abs/2212.13428), 2022
- [Retrieving and Reading: A Comprehensive Survey on Open-domain Question Answering](https://arxiv.org/abs/2101.00774), 2021
- [Knowledge Enhanced Pretrained Language Models: A Compreshensive Survey](https://arxiv.org/abs/2110.08455), 2021


### Tools

- [LangChain](https://github.com/langchain-ai/langchain): a framework for developing applications powered by language models. 
- [ChatGPT plugins](https://openai.com/blog/chatgpt-plugins): designed specifically for language models with safety as a core principle, and help ChatGPT access up-to-date information, run computations, or use third-party services.
- [EasyEdit](https://github.com/zjunlp/EasyEdit): an Easy-to-use Knowledge Editing Framework for LLMs.
- [FastEdit](https://github.com/hiyouga/FastEdit): injecting fresh and customized knowledge into large language models efficiently using one single command.
- [PyContinual](https://github.com/ZixuanKe/PyContinual): an Easy and Extendible Framework for Continual Learning.
- [Avalanche](https://github.com/ContinualAI/avalanche): an End-to-End Library for Continual Learning based on PyTorch.

## 🚩 Citation

If our research helps you, please kindly cite our paper.


## 🎉 Acknowledgement & Contribution

This field is evolving very fast, and we may miss important works. Please don't hesitate to share your work.
Pull requests are always welcome if you spot anything wrong (e.g., broken links, typos, etc.) or share new papers! 
We thank all contributors for their valuable efforts.



