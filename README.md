# Next Word Prediction using LSTM

A deep learning project that predicts the next word in a sequence using LSTM (Long Short-Term Memory) neural networks. The model is trained on Shakespeare's Hamlet and deployed as an interactive web application using Streamlit.

## Overview

This project demonstrates an end-to-end NLP pipeline for next word prediction:
- **Data Collection**: Shakespeare's Hamlet text from NLTK Gutenberg corpus
- **Text Preprocessing**: Tokenization and sequence generation
- **Model Training**: LSTM-based neural network
- **Deployment**: Interactive Streamlit web application

## Project Structure

```
.
|-- Training.ipynb          # Jupyter notebook for model training
|-- app.py                  # Streamlit web application
|-- hamlet.txt              # Training data (Shakespeare's Hamlet)
|-- next_word_prediction_lstm.h5  # Trained model weights
|-- tokenizer.pickle        # Saved tokenizer
|-- requirements.txt        # Python dependencies
|-- README.md               # Project documentation
```

## Model Architecture

The model uses a Sequential architecture with the following layers:

| Layer | Type | Output Shape | Parameters |
|-------|------|--------------|------------|
| 1 | Embedding | (None, 13, 100) | 481,800 |
| 2 | LSTM | (None, 13, 150) | 150,600 |
| 3 | Dropout (0.3) | (None, 13, 150) | 0 |
| 4 | LSTM | (None, 100) | 100,400 |
| 5 | Dropout (0.3) | (None, 100) | 0 |
| 6 | Dense (softmax) | (None, 4818) | 486,618 |

**Total Parameters**: ~1.2 million

## Training Details

- **Dataset**: Shakespeare's Hamlet (~4,818 unique words)
- **Sequence Length**: 13 words
- **Train/Test Split**: 80/20
- **Training Samples**: 20,585
- **Test Samples**: 5,147
- **Epochs**: 175
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Final Accuracy**: ~54%

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yokeswarans/End-to-End-LSTM-GRU---Predicting-the-next-word.git
cd End-to-End-LSTM-GRU---Predicting-the-next-word
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

Start the Streamlit app:
```bash
streamlit run app.py
```

This will open a web interface where you can:
1. Enter a sequence of words
2. Click "Predict Next Word"
3. View the predicted next word

### Training the Model (Optional)

To retrain the model, open and run `Training.ipynb` in Jupyter Notebook or Google Colab.

## Dependencies

- tensorflow==2.15.0
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorboard
- streamlit
- scikeras
- nltk

## Example

**Input**: "to be or not to be"

**Predicted Next Word**: The model will predict the most likely next word based on the training data.

## How It Works

1. **Text Preprocessing**: The input text is tokenized and converted to sequences of integers
2. **Padding**: Sequences are padded to match the model's expected input length
3. **Prediction**: The LSTM model outputs probability distribution over all words
4. **Output**: The word with highest probability is returned as prediction

## Future Improvements

- [ ] Train on larger and more diverse datasets
- [ ] Experiment with GRU layers
- [ ] Add beam search for multiple word suggestions
- [ ] Implement attention mechanism
- [ ] Deploy on cloud platform

## License

This project is open source and available for educational purposes.

## Author

**Yokeswaran S**

---

*Built with TensorFlow and Streamlit*
