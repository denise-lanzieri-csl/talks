# **Decoding the Material Foundation of Sustainable Technology:** A LLM-based classification of critical raw material functions in patent data

**Francesco de Cunzo¹, [Aurelio Patelli](https://github.com/aureliopatelli)¹, Angelica Sbardella¹, [Matteo Straccamore](https://github.com/MatteoStraccamore)¹², [Denise Lanzieri](https://github.com/dlanzieri)²**  
¹ [Enrico Fermi Research Center](https://www.cref.it/), Rome, Italy  
² [Sony Computer Science Laboratories – Rome](https://csl.sony.it/) (Sony CSL – Rome), Italy

---

## 🔬 Project Overview

This project introduces a **scalable and interpretable methodology** to map the role of Critical Raw Materials (CRMs) in patent-based innovation and their alignment with the **United Nations Sustainable Development Goals (SDGs)**. By combining **semantic similarity**, **manual annotation**, and **transformer-based classification**, we offer a **function-sensitive framework** to trace how materials enable or constrain sustainability-oriented technological development.

### 🎯 **Aim**  
To classify CRM-related patents by the **functional role** played by the material — whether it is *used*, *refined*, *recycled*, or *removed* — and connect this functional information to broader innovation patterns, technological domains, and policy-relevant sustainability targets.


### 🌍 **Scope**  
- Covers **566,587 CRM–patent associations** from EPO and WIPO (2000–2024)  
- Focuses on **31 materials** from the **2023 Critical Raw Materials Act** (EU)  
- Tracks trends across functions, technologies (CPC), countries, SDGs, and time

### 🧩 **Methodology**
1. **Semantic matching between CPC classes and SDGs**  
   Cosine similarity between CPC subclass titles and SDG targets using Sentence Transformers.

2. **Keyword search of CRMs in patent abstracts**  
   Extraction of CRM–abstract pairs via filtered keyword search on 500k+ patents.

3. **Assignment of CRM function categories**  
   Manual classification into five roles: *use*, *refine*, *recycle*, *remove*, or *wrong* (false positive).

4. **Classification with fine-tuned LLM**  
   Fine-tuning a BERT-based model on 11,500 labeled examples, reaching **94% accuracy**.

### 🧭 **Impact & Policy Relevance**
- Enables **country-level monitoring** of CRM innovation strategies (e.g., focus on *recycling* vs. *use*)
- Assesses maturity of **circular economy pathways**
- Identifies **innovation bottlenecks** and **functional gaps** by material and domain

---

## 🌐 **Key Resources**

> 🔗 **🌟 VISIT THE INTERACTIVE DASHBOARD**  
> 👉 [**https://sites.google.com/cref.it/material-decoders/home?authuser=1**](https://sites.google.com/cref.it/material-decoders/home?authuser=1)  
> *(Core to the project's visibility and usability)*

> 📄 **Read the full project report:**  
> 👉 [Presentation and Supporting Documents/Report.pdf](./Presentation%20and%20Supporting%20Documents/Report.pdf)  
> *(Detailed description of data, methods, architecture, and results)*

---
## 📁 Repository Structure

```
CFS25-Material-Decoders/
├── Presentation and Supporting Documents/    
│   ├── Report.pdf
├── Source Code/
│   ├── data/
│   ├── outputs/      
│   ├── Notebook/                             
│   │   ├── TextSimilarity_CPC_SDGs.ipynb
│   │   └── Download_Patent_Data.ipynb
│   │   ├── tokenization_4_fine_tuning_1.ipynb
│   │   ├── TextMining_and_CollectingMetadata.ipynb
│   │   └── tokenization_4_fine_tuning_2.ipynb
│   ├── scripts/                               
│   │   ├── fine_tuning_1.py
│   │   └── fine_tuning_2.py
│   │   └── download_data_inference.sh
│   │   └── download_data_training.sh
│   │   └── predict_dataset_labels.py
│   └── EPO_CodeFest_2025/                   
│       └── prepare_data.py                   
├── Supporting Materials and Tools/           
│   └── html_figures/                   
│       └── *.html                   
├── README.md
```

# 📦 Patent Classification Fine-Tuning Pipeline

This repository provides a complete pipeline for fine-tuning BERT-style models on patent abstract data using compressed CPC codes as multi-label targets. It supports scalable processing and training via HuggingFace Transformers.


---


## 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone git@github.com:epo/CFS25-Material-Decoders.git
cd Source\ Code/scripts/
```


## 2. ⚙️ Preprocessing

###   2.1 Text Similarity between CPC codes and UN SDGs 

Run the notebook [TextSimilarity_CPC_SDGs.ipynb](https://github.com/epo/CFS25-Material-Decoders/blob/denise/Source%20Code/notebooks/TextSimilarity_CPC_SDGs.ipynb).

This module computes semantic similarity between CPC technology codes and Sustainable Development Goals (SDGs) by leveraging the `all-mpnet-base-v2` SentenceTransformer. Official CPC and SDG texts are encoded into embeddings, and cosine similarity is used to identify the most relevant SDGs for each technology. The result is a ranked mapping between technological domains and sustainability goals, useful for downstream analysis and policy insights.

###   2.2 Patent Dataset Download and Saving 

Run the notebook [Download_Patent_Data.ipynb](https://github.com/epo/CFS25-Material-Decoders/blob/denise/Source%20Code/notebooks/Download_Patent_Data.ipynb).

This notebook will create the patent dataset needed for the first step of the fine tuning. The dataset we used is downloaded in the next step
(see 3. Download the Dataset). The goal is to build a file of approximately 8 million patents with patent abstract and CPC codes. The patents are randomly extracted. 

In particular, the notebook tries to select 2.5 million patents (with English abstract and CPC codes) for each of the four 6-year ranges 2000-2005; 2006-2011; 2012-2017; 2018-2023. This would yield a sample of 10 million patents. However, due to missing patent data–i.e. missing abstract and/or CPC codes–the number of selected patents is lower. In particular, the sample we used for the analysis contains 7_619_400 patents. 
The selection of the four time intervals is driven by (i) the focus period of our analysis (2000 onwards) and (ii) the lower patent volumes in the last year of PATSTAT due to lags in registering recent patents

###  2.3 Text Mining and Collecting Metadata 

Run the notebook [TextMining_and_CollectingMetadata.ipynb](https://github.com/epo/CFS25-Material-Decoders/blob/denise/Source%20Code/notebooks/TextMining_and_CollectingMetadata.ipynb).

This notebook performs the keyword search pre-filtering of the patents in PATSTAT in order to identify the critical raw materials (CRM) related patents–i.e. the patents with at least one mention of a CRM associated keyword. Additionally, once the CRM-related patents are identified, we collect metadata on CPC codes and countries of inventors and applicants. 

The PATSTAT tables we use are TLS201_APPLN–with information on patent application ids, earliest filing years and patent offices among others–TLS203_APPLN_ABSTR–containing the patent application ids, their abstracts, and the language in which they are written–TLS206_PERSON and TLS207_PERS_APPLN–for the countries of origin of inventors and applicants–and TLS224_APPLN_CPC–for CPC codes.

This part of the analysis can be summarised into 3 steps. In the first step, we collect all patents filed in EPO and WIPO patent offices between 2000 and 2025. This is the starting dataset. In the second step, we perform the keyword search investigation, looking for CRMs in the patent abstracts retrieved from TLS203_APPLN_ABSTR. In the third step, we collect metadata on CPC codes and countries of inventors, applicants, and patent offices from TLS206_PERSON, TLS207_PERS_APPLN, and TLS224_APPLN_CPC


## 3. 🧹 Prepare the Data for Training

### 3.1 Download Training data
Run the following shell [script](https://github.com/epo/CFS25-Material-Decoders/blob/denise/Source%20Code/scripts/download_data_training.sh) to download the necessary datasets:
```bash
bash download_data_training.sh
```
### 3.2 Tokenize Training Data
Run the following notebooks in order to clean and tokenize the dataset:
- [tokenization_4_fine_tuning_1.ipynb](https://github.com/epo/CFS25-Material-Decoders/blob/denise/Source%20Code/notebooks/tokenization_4_fine_tuning_1.ipynb): 
  - Cleans patent abstracts  
  - Compresses CPC codes (e.g., `H01M10` → `[H01M10_token]`)  
- [tokenization_4_fine_tuning_2.ipynb](https://github.com/epo/CFS25-Material-Decoders/blob/denise/Source%20Code/notebooks/tokenization_4_fine_tuning_2.ipynb): 
  - Tokenize abstracts with `BertTokenizerFast`
  - Save the dataset into manageable chunks
  - Ensure the `TRAIN_MODE` flag is set to `True`

## 4. 🧪 Fine-Tune the Model

The fine-tuning process lies at the heart of our approach to developing a classifier capable of identifying the **specific functional roles of Critical Raw Materials (CRMs)** in patented inventions. These functional roles include: “use”, “recycle”, “refine”, “remove”, and “wrong” (irrelevant mentions). This classification allows us to go beyond simple keyword-based searches and truly decode how CRMs enable or are targeted by technological innovation.

Our methodology consists of two sequential training stages:

### 🔁 Stage 1 – Domain-Specific Pre-Training 

```bash
python scripts/fine_tuning_1.py
```

In this stage (see script [here](https://github.com/epo/CFS25-Material-Decoders/blob/denise/Source%20Code/scripts/fine_tuning_1.py)), we train a masked language model (MLM) based on a pre-trained architecture (e.g., BERT for chemicals) using a large corpus of **unlabeled patent abstracts** from the years 2000–2023. This step allows the model to specialize in the language and structure of patent texts, capturing domain-specific terminology and context relevant to technological and material innovation.

### 🔄 Stage 2 – Supervised Fine-Tuning on Annotated CRM Data 

```bash
python scripts/fine_tuning_2.py
```

In the second stage (see script [here](https://github.com/epo/CFS25-Material-Decoders/blob/denise/Source%20Code/scripts/fine_tuning_2.py)), we load the domain-specific model from Stage 1 and fine-tune it on a **manually annotated dataset** of CRM-related patent abstracts. Each abstract is labeled with the functional role of the CRM it mentions. Crucially, this model does not rely only on the abstract content: it also incorporates **Cooperative Patent Classification (CPC) codes** as additional features to enrich the contextual understanding of the invention.

This supervised training step enables the model to learn subtle distinctions in how materials are mentioned and applied, improving classification accuracy and ensuring strong generalization to the broader patent corpus.

Together, these two stages result in a robust, function-specific classifier that offers high-quality insights into how strategic raw materials contribute to innovation—supporting evidence-based decision-making on sustainability, technology policy, and industrial strategy.

The fine-tuning process is central to our approach for building a classifier that identifies **the specific functional roles played by Critical Raw Materials (CRMs)** in patented technologies. These roles include whether a CRM is being used in the invention, targeted for recycling or refinement, removed from the process, or falsely mentioned without relevance (“wrong”). Accurately capturing these distinctions is essential to understand the real technological significance of CRMs and their alignment with key UN Sustainable Development Goals (SDGs).


## 5. 🔎 Prediction
### 5.1 Download Data for Inference
Run the following shell [script](https://github.com/epo/CFS25-Material-Decoders/blob/denise/Source%20Code/scripts/download_data_inference.sh) to download a set of unlabeled patents for which the model will predict the functional role of CRMs:

```bash
bash download_data_inference.sh
```
Places files in:  `../EPO_CodeFest_2025/data/`.

### 5.2  Tokenize Inference Data
To prepare the new, unlabeled data for prediction, you need to re-run the tokenization notebook used during training:

1. Open the notebook: [tokenization_4_fine_tuning_2.ipynb](https://github.com/epo/CFS25-Material-Decoders/blob/denise/Source%20Code/notebooks/tokenization_4_fine_tuning_2.ipynb)
2. **Set the following flag to inference mode:**

   ```python
   TRAIN_MODE = False  # Important: switch to inference mode


### 5.3 Run Predictions
Execute the [script](https://github.com/epo/CFS25-Material-Decoders/blob/denise/Source%20Code/scripts/predict_dataset_labels.py) below to predict the functional roles of the CRMs:
```bash
python scripts/predict_dataset_labels.py
```
Alternatively, you can download the predicted results—the outcome of this pipeline—directly from this [link](https://drive.google.com/file/d/1hVbqnRqSCapczCWmhpuGVS5TGwpQxfxe/view?usp=sharing).

## 🔧 Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `tensorflow`
- `transformers`
- `datasets`
- `pandas`
- `scikit-learn`
- `tokenizers`
- `keras`
- `fitz`
- `sentence_transformers`
- `torch`


## 🧠 CPC Token Compression Logic

CPC codes are processed as follows:
- Compressed to **6-digit** prefixes (e.g., `H01M10`)
- Encoded with `_token` suffix (e.g., `[H01M10_token]`)
- Exceptions: CPCs starting with `Y02` or `Y04` are retained in full.

## 📊 Output

- Fine-tuned BERT models compatible with HuggingFace format
- Tokenized datasets
- Intermediate and final training checkpoints (optional)
