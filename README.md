# Home-Assignment-4

Name:Sushma Veluru

700#: 700765390

CRN: 23848

# NLP Project Tasks

This repository includes implementations of foundational Natural Language Processing (NLP) tasks using Python and popular libraries such as NLTK, spaCy, NumPy, and HuggingFace Transformers.

---

## Q1: NLP Preprocessing Pipeline

### Task Description
Perform basic NLP preprocessing steps on a sentence:
1. Tokenization
2. Stopword Removal
3. Stemming

**Input Sentence:**
```
"NLP techniques are used in virtual assistants like Alexa and Siri."
```

### Expected Output
```
Original Tokens: ['NLP', 'techniques', 'are', 'used', 'in', 'virtual', 'assistants', 'like', 'Alexa', 'and', 'Siri', '.']
Tokens Without Stopwords: ['NLP', 'techniques', 'used', 'virtual', 'assistants', 'like', 'Alexa', 'Siri']
Stemmed Words: ['nlp', 'techniqu', 'use', 'virtual', 'assist', 'like', 'alexa', 'siri']
```

---

## Q2: Named Entity Recognition with spaCy

### Task Description
Use spaCy to extract named entities from a sentence. Print:
- Entity text
- Entity label (e.g., PERSON, DATE)
- Start and end character positions

**Input Sentence:**
```
"Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."
```

### Expected Output
```
Entity: Barack Obama | Label: PERSON | Start: 0 | End: 12
Entity: 44th | Label: ORDINAL | Start: 24 | End: 28
Entity: United States | Label: GPE | Start: 46 | End: 60
Entity: Nobel Peace Prize | Label: ORG | Start: 74 | End: 92
Entity: 2009 | Label: DATE | Start: 96 | End: 100
```

---

## Q3: Scaled Dot-Product Attention

### Task Description
Implement the scaled dot-product attention mechanism using NumPy:
1. Compute Q × Kᵀ
2. Scale by √d
3. Apply softmax
4. Multiply by V to get final output

**Input:**
```python
Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
```

### Expected Output
```
Attention Weights Matrix: [[0.8808 0.1192], [0.1192 0.8808]]
Final Output Matrix: [[1.4768 2.4768 3.4768 4.4768], [4.5232 5.5232 6.5232 7.5232]]
```

---

## Q4: Sentiment Analysis with HuggingFace Transformers

### Task Description
Use a pre-trained sentiment analysis model from HuggingFace to classify a sentence.

**Input Sentence:**
```
"Despite the high price, the performance of the new MacBook is outstanding."
```

### Expected Output
```
Sentiment: POSITIVE
Confidence Score: 0.9992
```

---

## Requirements
- Python 3.7+
- `nltk`
- `spacy`
- `numpy`
- `transformers`
- `torch`

Install dependencies with:
```bash
pip install nltk spacy numpy transformers torch
python -m nltk.downloader stopwords punkt
python -m spacy download en_core_web_sm


