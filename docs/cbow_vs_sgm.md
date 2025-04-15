# Efficient Estimation of Word Representations in Vector Space (Word2Vec) 📖

### Definition 📌
This paper introduces two architectures to efficiently learn word embeddings—vectors that capture semantic and syntactic relationships between words—from very large text datasets.

- **Continuous Bag-of-Words (CBOW)**: Predicts the current word using context (surrounding words).
- **Continuous Skip-gram**: Predicts surrounding words given the current word.

---

## Motivation and Goal 🎯
Traditionally, NLP models treat words as independent units without semantic similarity. Mikolov's models capture word similarity and relationships mathematically.

**Goals:**
- Efficiently learn high-quality word vectors from very large datasets (billions of words).
- Preserve linear regularities (relationships) among words.

---

## Analogy 💡
Think of word embeddings as positioning words in a large space (like planets in space), where related words orbit closely, and relationships between words can be represented as directions or vectors between planets (words).

Example:
- **Vector("King") - Vector("Man") + Vector("Woman") ≈ Vector("Queen")**

---

## Model Architectures 🧩

### 1. Continuous Bag-of-Words (CBOW) 🧮
Predicts the middle word using surrounding words as input. It simplifies neural networks by removing the non-linear hidden layer, averaging context word vectors directly.

**Formula for computational complexity (training cost):**

$$
Q = N \times D + D \times \log_2(V)
$$

- **N**: number of context words
- **D**: dimensionality of word vectors
- **V**: vocabulary size

### Diagram 🔍
```
Context words --> Projection (average) --> Predicted word
```

**Example**:  
Predict **"fox"** from context words (**"quick", "brown", "jumps", "over"**).

---

### 2. Continuous Skip-gram 🧮
Predicts context words from a given target word. It focuses on predicting multiple words within a certain range around a given input word.

**Formula for computational complexity:**
$$
Q = C \times (D + D \times \log_2(V))
$$

- **C**: context window size around a target word.

### Diagram 🔍
```
Input word --> Projection --> Context words predictions
```

**Example**:  
Given **"fox"**, predict context (**"quick", "brown", "jumps", "over"**).

---

## Comparing Architectures 🆚

| Architecture  | Task         | Accuracy  | Training Speed 🚀 |
|---------------|--------------|-----------|-------------------|
| NNLM          | Semantic     | Moderate  | Slow 🐢           |
| RNNLM         | Syntactic    | Moderate  | Very Slow 🐢🐢     |
| **CBOW**      | Syntactic    | High ✅   | Fast 🐇            |
| **Skip-gram** | Semantic     | High ✅   | Moderate 🐎        |

**Takeaway**:  
- CBOW is faster to train and effective for syntactic tasks.  
- Skip-gram captures semantic relationships better.

---

## Practical Steps to Implement CBOW 🔄

### Step 1: Tokenization & Vocabulary Creation 📚
- Tokenize corpus into words (e.g., splitting by space, removing punctuation).
- Build vocabulary: frequency-based, select top-N words (e.g., top 30k words).
- Assign unique indices to each word.

**Example Python Code** 🛠️:
```python
from collections import Counter
import re

text = "The quick brown fox jumps over the lazy dog"
tokens = re.findall(r'\w+', text.lower())

# Building vocabulary
vocab_count = Counter(tokens)
vocab = {word: i for i, (word, _) in enumerate(vocab_count.items())}
print(vocab)
```

---

### Step 2: Training Data Generation 🧑‍💻
- For each target word, take surrounding context words within a certain window.

**Example**:
```
Context size = 2, sentence: "quick brown [fox] jumps over"
CBOW Input: ["quick", "brown", "jumps", "over"]
CBOW Output: "fox"
```

---

### Step 3: Model Implementation (PyTorch) 🛠️

**CBOW Model Definition**
```python
import torch
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        context_emb = torch.mean(embeds, dim=1)
        out = self.linear(context_emb)
        return out
```

---

### Step 4: Training 🔄
- Use cross-entropy loss and an optimizer (e.g., Adam or SGD).

**Simple Training Loop**
```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for context, target in data_loader:
        optimizer.zero_grad()
        output = model(context)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
```

---

## Evaluating Word Vectors 📊
- Use semantic-syntactic analogy tasks.
- Evaluate vector arithmetic accuracy.

**Example Evaluation**:
```
vector("Paris") - vector("France") + vector("Italy") ≈ vector("Rome")
```

---

## Key Takeaways 🔥

- Simple architectures (CBOW, Skip-gram) efficiently produce high-quality embeddings.
- CBOW is computationally cheaper; Skip-gram excels at semantic relationships.
- High-dimensional vectors (300-600) trained on large data (1B+ words) yield better results.
- Vectors can solve analogy tasks through simple arithmetic operations.

---

## Next Steps for You 🎯

- Implement your tokenizer/vocabulary creation as above.
- Train CBOW on a sizable corpus.
- Experiment with embedding dimensions (e.g., 300-600).
- Evaluate using analogy tasks to test semantic/syntactic capabilities.

---

## Summary 🚩
Mikolov et al.’s paper introduces efficient neural architectures (CBOW & Skip-gram) for learning meaningful word embeddings. They demonstrate excellent semantic and syntactic capture with computational efficiency, making them ideal for large-scale NLP tasks.
