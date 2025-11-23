import streamlit as st  
import numpy as np  
import pickle 

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model=load_model('next_word_prediction_lstm.h5')

#Load the tokenizer
with open ('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)


#function to predict the next word
def predict_next_word(model,tokenizer,text,max_sequence_len):

    # Convert the text to sequences of integer
    token_list=tokenizer.texts_to_sequences([text])[0]

    # Trim the sequence if longer than allowed
    if len(token_list)>=max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):]

    # Pad to correct length
    token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre',)

    # Predict probability distribution
    predicted=model.predict(token_list,verbose=0)

    # Get highest probability index
    predicted_word_index=np.argmax(predicted,axis=1)

    # Map index → word
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None

#Streamlit app 
st.title("Next Word Prediction using LSTM")
input_text=st.text_input("Enter the sequence of words")
if st.button("Predict Next Word"):
    max_sequence_len=model.input_shape[1]+1 
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next word:{next_word}")