import streamlit as st
import preprocessing as pf
import joblib
import numpy as np
from scipy.special import softmax
import pandas as pd

# Load the tokenizer and model
bert_tokenizer = joblib.load("bert_tokenizer.pkl")
bert_model_path = r'twitter_ner_bert_best_model/best_model_weights.h5'
MAX_LEN = 35

# Load trained model
bert_loaded_model = pf.build_bert_model(max_len=MAX_LEN, tags_2_idx=bert_tokenizer['label2idx'])
bert_loaded_model.load_weights(bert_model_path)

# Streamlit UI
st.set_page_config(page_title="Named Entity Recognition", layout="wide")
st.title("🚀 Named Entity Recognition (NER) with BERT")
st.write("Enter a sentence below to detect named entities.")

# Sidebar for styling
st.sidebar.title("🔧 Settings")
st.sidebar.markdown("This app extracts named entities from input text using a trained BERT model.")

# User Input
inp = st.text_area("Enter a sentence:", "there is your dog let's go !!")

if st.button("Analyze Text"):
    with st.spinner("Processing... 🚀"):
        # Preprocess input
        sentence = pf.uncleaned_procure_datatset(inp)
        tt = pd.DataFrame({'sentence': [sentence]})
        tt = list(tt['sentence'].values)

        berttokenizer_test = pf.BertNERTokenizer_Test(tokenizer=bert_tokenizer, model_name='bert-base-uncased', max_len=MAX_LEN)
        bertclean_test_dataset = berttokenizer_test.transform(tt)

        X_test1 = np.array(list(bertclean_test_dataset["token_id"]))
        X_test2 = np.array(list(bertclean_test_dataset["segment_id"]))
        X_test3 = np.array(list(bertclean_test_dataset["attention_mask"]))

        # Make predictions
        y_pred_logits = bert_loaded_model.predict([X_test1, X_test2, X_test3])["potentials"]
        y_pred_logits = softmax(y_pred_logits, axis=-1)
        y_pred_indices = np.argmax(y_pred_logits, axis=-1)

        validation_predictions = pf.get_predicted_entities(
            all_prediction=y_pred_indices,
            sub_sentence=bertclean_test_dataset.padded_sentence.values,
            ignore_labels=[bert_tokenizer['label2idx']['O'], bert_tokenizer['label2idx']['<PAD>']]
        )

        sentence_tokens = [bert_tokenizer['idx2word'][idx] for idx in X_test1[0] if idx != bert_tokenizer['word2idx']["<PAD>"]]
        pred_labels = [bert_tokenizer['idx2label'][idx] for idx in y_pred_indices[0] if idx != bert_tokenizer['label2idx']["<PAD>"]]
        
        entities = [s for s in validation_predictions[0] if ("<PAD>" not in s) and (len(s) > 1)]

        # Display Results
        st.subheader("🔍 Results")
        st.write(f"**Cleaned Text:** {' '.join(sentence_tokens)}")
        st.write(f"**Predicted Labels:** {pred_labels}")
        st.write(f"**Entities Found:** {entities if entities else 'No named entities detected.'}")

        # Visualization
        st.subheader("📊 NER Visualization")
        pf.visualize_ner(sentence_tokens, pred_labels)

st.sidebar.markdown("🧠 Built with BERT and Streamlit")
st.sidebar.text("Author: Shubham 🚀")
