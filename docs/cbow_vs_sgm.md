# **Continuous Bag-of-Words (CBOW) and Skip-Gram Models: An In-Depth Explanation 📚🔍**

In natural language processing (NLP), word embeddings are a powerful way to represent words as dense vectors that capture semantic relationships. Two fundamental models for generating word embeddings are the **Continuous Bag-of-Words (CBOW)** model and the **Skip-Gram** model. Both models are components of the Word2Vec framework and are used to learn word representations from large text corpora. Below, we present a detailed explanation of each model, including their mathematical formulations, key differences, and visualizations.

---

## **1️⃣ Overview of Word2Vec Models**

Word2Vec models learn word embeddings by examining a large corpus of text. Their goal is to assign vectors to words such that words that appear in similar contexts end up with similar vectors. The two primary approaches are:

- **Continuous Bag-of-Words (CBOW):**  
  Predicts the target word from its surrounding context (neighboring words).

- **Skip-Gram:**  
  Predicts surrounding context words given the target word.

---

## **2️⃣ Continuous Bag-of-Words (CBOW) Model**

### **How CBOW Works**

- **Idea:**  
  Given a set of context words surrounding a target word, the model predicts the target word.  
  For example, in the sentence "The **quick** brown fox jumps over the lazy dog", if the context is defined as two words before and after the target word, then for the target word "**brown**", the context words would be:  
  $$
  [\text{"The"}, \text{"quick"}, \text{"fox"}, \text{"jumps"}]
  $$

- **Architecture:**  
  - **Input Layer:** The context words are represented as one-hot vectors (or indices) and then projected to a common embedding space using an embedding matrix.  
  - **Projection Layer:** Averages (or sums) the embeddings of the context words.  
  - **Output Layer:** Uses a softmax function to predict the probability distribution over the vocabulary for the target word.

### **Mathematical Formulation**

1. **Input:**  
   Let the context window size be $ C $, and let the context words be $ w_{t-C}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+C} $.

2. **Embedding:**  
   Each context word $ w $ is represented as an embedding vector $ v_{w} $.  
   The average context vector is computed as:
   $$
   v_{\text{context}} = \frac{1}{2C} \sum_{\substack{-C \leq j \leq C \\ j \neq 0}} v_{w_{t+j}}
   $$

3. **Output (Softmax):**  
   The probability of predicting the target word $ w_t $ is given by:
   $$
   p(w_t|w_{t-C}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+C}) = \frac{\exp(v'_{w_t} \cdot v_{\text{context}})}{\sum_{w \in V} \exp(v'_{w} \cdot v_{\text{context}})}
   $$
   - $ v'_{w} $ are the output (context) embeddings.
   - $ V $ is the vocabulary.

---

### **Visualization of the CBOW Model**

Below is a diagram (in Mermaid syntax) representing the CBOW model:

```mermaid
flowchart TD
    subgraph Context_Words [Context Words]
        A1[Word $begin:math:text$w_{t-C}$end:math:text$]
        A2[Word $begin:math:text$w_{t-C+1}$end:math:text$]
        A3[...]
        A4[Word $begin:math:text$w_{t-1}$end:math:text$]
        A5[Word $begin:math:text$w_{t+1}$end:math:text$]
        A6[...]
        A7[Word $begin:math:text$w_{t+C}$end:math:text$]
    end

    subgraph Embedding [Embedding Layer]
        E1[Embedding for $begin:math:text$w_{t-C}$end:math:text$]
        E2[Embedding for $begin:math:text$w_{t-C+1}$end:math:text$]
        E3[...]
        E4[Embedding for $begin:math:text$w_{t-1}$end:math:text$]
        E5[Embedding for $begin:math:text$w_{t+1}$end:math:text$]
        E6[...]
        E7[Embedding for $begin:math:text$w_{t+C}$end:math:text$]
    end

    subgraph Average [Averaging]
        AVG[Average Embedding $begin:math:text$v_{context}$end:math:text$]
    end

    subgraph Output [Softmax Output]
        OUT[Predict $begin:math:text$w_t$end:math:text$]
    end

    A1 --> E1
    A2 --> E2
    A3 --> E3
    A4 --> E4
    A5 --> E5
    A6 --> E6
    A7 --> E7
    E1 ---|
    E2 ---| 
    E3 ---|--> AVG
    E4 ---|
    E5 ---|
    E6 ---|
    E7 ---|
    AVG --> OUT
```

•	Explanation:
•	Context words are first transformed into embedding vectors.
•	These embeddings are averaged to form a single context representation.
•	The average vector is then used to predict the target word using a softmax layer.

---
## 3️⃣ Skip-Gram Model

How Skip-Gram Works
	•	Idea:
The Skip-Gram model takes a target word and uses it to predict its surrounding context words.
For instance, given the target word “brown” in the sentence “The quick brown fox jumps over the lazy dog”, the model attempts to predict the surrounding words:
[
[\text{“The”}, \text{“quick”}, \text{“fox”}, \text{“jumps”}]
]
	•	Architecture:
	•	Input Layer: The target word is represented as a one-hot vector and then projected to an embedding space.
	•	Projection Layer: The embedding for the target word is used directly.
	•	Output Layer: For each context position, a softmax is computed to predict the corresponding context word.

Mathematical Formulation
	1.	Input:
Let the target word be ( w_t ).
	2.	Embedding:
The target word is represented as its embedding vector ( v_{w_t} ).
	3.	Output (Softmax for Each Context Position):
For each context word ( w_{t+j} ) (where ( j ) runs over the context positions), predict:
$$
p(w_{t+j}|w_t) = \frac{\exp(v’{w{t+j}} \cdot v_{w_t})}{\sum_{w \in V} \exp(v’{w} \cdot v{w_t})}
$$
	•	( v’_{w} ) are the output embeddings for each word.
	•	This prediction is performed for all context words simultaneously (or independently).

⸻

Visualization of the Skip-Gram Model

Below is a Mermaid diagram representing the Skip-Gram model:

flowchart TD
    T[Target Word $begin:math:text$w_t$end:math:text$]
    subgraph Embedding_Skip [Embedding Layer]
        E[Embedding $begin:math:text$v_{w_t}$end:math:text$]
    end

    subgraph Prediction [Skip-Gram Outputs]
        P1[Predict $begin:math:text$w_{t-C}$end:math:text$]
        P2[Predict $begin:math:text$w_{t-C+1}$end:math:text$]
        P3[...]
        P4[Predict $begin:math:text$w_{t-1}$end:math:text$]
        P5[Predict $begin:math:text$w_{t+1}$end:math:text$]
        P6[...]
        P7[Predict $begin:math:text$w_{t+C}$end:math:text$]
    end

    T --> E
    E --> P1
    E --> P2
    E --> P3
    E --> P4
    E --> P5
    E --> P6
    E --> P7

	•	Explanation:
	•	The target word (w_t) is transformed into its embedding (v_{w_t}).
	•	This embedding is used to predict each of the context words simultaneously through the softmax layers.
	•	The model is designed to maximize the probability of the correct context words given (w_t).

⸻

4️⃣ Key Differences between CBOW and Skip-Gram

Aspect	CBOW	Skip-Gram
Input/Output	Inputs: Context words; Output: Target word	Input: Target word; Outputs: Context words
Focus	Learns an overall representation by averaging context	Learns to predict surrounding words from a single target
Efficiency	More efficient with frequent words	Better for infrequent words
Training Complexity	Simpler to train when context is well-defined	Can be more computationally intensive due to multiple predictions



⸻

5️⃣ Practical Considerations
	•	Training Objectives:
	•	Both models aim to maximize the likelihood of observed word-context pairs.
	•	The loss function typically used is the negative log-likelihood of the predicted word given the context (or vice versa), optimized using stochastic gradient descent or other advanced optimizers.
	•	Use Cases:
	•	CBOW tends to perform well when the context window is small and when dealing with frequent words.
	•	Skip-Gram can perform better with small datasets or for infrequent words, as it focuses more on individual target words.

⸻

🚀 Final Takeaways
	1.	CBOW Model:
	•	Predicts a target word based on the surrounding context words.
	•	Uses an average of context embeddings to produce a prediction.
	2.	Skip-Gram Model:
	•	Predicts the context words based on a single target word.
	•	Uses the target word embedding to generate multiple predictions.
	3.	Inductive Bias:
	•	Both models assume that word meanings can be captured through the distributional hypothesis (“You shall know a word by the company it keeps”).
	4.	Applications:
	•	Used for learning word embeddings that capture semantic relationships, which are foundational for many NLP tasks like text classification, translation, and sentiment analysis.
	5.	Visualizations & Diagrams:
	•	Diagrams help illustrate how the data flows through each model, ensuring a clear understanding of their mechanisms.

This comprehensive explanation, along with diagrams and examples, should give you a solid understanding of both the CBOW and Skip-Gram models. If you have any questions or need further clarifications, feel free to ask! 😊🔥

