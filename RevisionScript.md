# LSTM Text Generation Model — Revision Notes

## 1. Embedding Layer
- Input: 13 word IDs  
- Output: (13 × 100) matrix  
- Each word ID becomes a 100-dimensional **learned meaning vector**.  
- Embedding weights **are trainable** and get updated during backprop.

---

## 2. First LSTM Layer (150 units)
- Input: (13 × 100)  
- Output: (13 × 150)  
- This LSTM processes **each of the 13 words in sequence** and creates a 150-dimensional **context vector** for each timestep.
- Meaning of each vector:
  - “word meaning + what came before it”
- 150 units = 150 LSTM cells → each cell performs:
  - forget gate  
  - input gate  
  - candidate  
  - output gate  

Why 150 > 100?  
- It’s common to expand the dimensionality so the model can encode **more sequence information** than original embeddings.

---

## 3. Second LSTM Layer (100 units)
- Input: (13 × 150)  
- Output: a single (100) vector  
- This LSTM reads the full 13-step sequence and returns **only the final hidden state**.
- This produces a 100-dimensional **summary of the entire sequence**.

How compression happens:  
- Internal recurrence folds all 13 context vectors into the final hidden state.  
- The 100-dim final vector captures the overall meaning of all 13 inputs.

---

## 4. Dense Softmax Layer
- Input: 100-dim summary vector  
- Output: 4818-dim vector (probabilities for each word in vocabulary)  
- Dense layer performs:

- This chooses the **next word**.

---

## 5. Why Two LSTMs?
- First LSTM (150 units): learns **per-word contextual meaning** across the sequence.  
- Second LSTM (100 units): compresses the entire 13-step sequence into **one final understanding vector** used for prediction.  
- Stacking LSTMs increases model depth → richer understanding of patterns, grammar, and dependencies.

---

## 6. Overall Data Flow Diagram (Simple)

Word IDs (13)
↓
Embedding (13 × 100)
↓
LSTM-1 (150 units) → (13 × 150)
↓
LSTM-2 (100 units) → (100)
↓
Dense Softmax (4818)
↓
Next-word prediction


---

## 7. Key Intuitions
- Embedding gives meaning to IDs.  
- LSTM-1 extracts context for every word.  
- LSTM-2 compresses the whole sentence.  
- Dense predicts the next word.  
- Two LSTMs = deeper understanding of sequence patterns.

---



