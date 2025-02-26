#BERT preprocessing task

import pandas as pd
import numpy as np
import contractions
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from transformers import TFBertForTokenClassification
from keras.saving import register_keras_serializable
from transformers import BertTokenizer as bt

def uncleaned_procure_datatset(text):
    text=text.split()
    t=[]
    for word in text:
        word=word.strip().lower()
        if word=="'":
            t.append(word)
        else:
            if word[0]=="'" and len(t)>0:
                word = [contractions.fix(t[-1]+word).split()[-1]]
                for w in word:
                    t.append(w)
            else:
                word=contractions.fix(word).split()
                for w in word:
                    t.append(w)
    
    return t

    

@register_keras_serializable()
class TFBertLayer(Layer):
    def __init__(self, model_name, num_labels, **kwargs):
        super(TFBertLayer, self).__init__(**kwargs)
        self.model_name = model_name
        self.num_labels = num_labels
        self.bert = TFBertForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels, from_pt=False
        )
        

    def build(self, input_shape):
        """ Ensures that trainable weights are properly registered in TensorFlow """
        self.trainable_weights.extend(self.bert.trainable_variables)
        self.non_trainable_weights.extend(self.bert.non_trainable_variables)
        super().build(input_shape)

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs.logits  # Extract logits


    def get_config(self):
        config = super().get_config()
        config.update({
            "model_name": self.model_name,
            "num_labels": self.num_labels
        })
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_bert_model(max_len, tags_2_idx):


    # Define model inputs
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    token_type_ids = Input(shape=(max_len,), dtype=tf.int32, name="token_type_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")


    # Use custom wrapper layer
    bert_output = TFBertLayer("bert-base-uncased", num_labels=len(tags_2_idx))([input_ids, attention_mask, token_type_ids])


    # Define final model
    model = Model(inputs=[input_ids, token_type_ids, attention_mask], outputs={"potentials": bert_output})


    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


    # Compile model
    model.compile(optimizer=Adam(learning_rate=3e-5), loss={"potentials": loss}, metrics={"potentials": ['accuracy']})


    return model

## output is raw logits -- need to convert them to softmax
class BertNERTokenizer_Test:
    def __init__(self, tokenizer, model_name='bert-base-uncased', max_len=30): 
        self.tokenizer = bt.from_pretrained(model_name)
        self.word2idx = tokenizer['word2idx']
        self.label2idx = tokenizer['label2idx']
        self.idx2word = tokenizer['idx2word']
        self.idx2label = tokenizer['idx2label']
        self.max_len = max_len
        self.n_tags = len(self.label2idx)

    def tokenize(self, sentence):
        tokens= []

        for word in sentence:
            tokenized_word = self.tokenizer.tokenize(word)  # Tokenize using BERT
            tokens.extend(tokenized_word)
            #new_labels.extend([label] + [label] * (len(tokenized_word) - 1))  # Align labels

        return tokens

    def transform(self, dataset):
        sentences= []

        for i in tqdm(range(len(dataset)),desc='Cleaned Test Text', total= len(dataset)):
            sentence=dataset[i]
            #label=dataset[i][1]
            tokenized_sentence = self.tokenize(sentence)
            sentences.append(tokenized_sentence)
            #labels.append(tokenized_labels)

        # Convert tokens to input IDs
        X_test_token_ids = [[self.word2idx.get(w, self.word2idx["<UNK>"]) for w in s] for s in sentences]

        # Padding sequences
        X_test_token_ids = pad_sequences(maxlen=self.max_len, sequences=X_test_token_ids, padding="post", value=self.word2idx["<PAD>"])

        # Convert labels to label IDs
        #X_test_labels = [[self.label2idx[w] for w in s] for s in labels]

        # Padding labels
        #X_test_labels = pad_sequences(maxlen=self.max_len, sequences=X_test_labels, padding="post", value=self.label2idx["<PAD>"])

        X_test_padded_sent= [sent+["<PAD>"]*(0 if self.max_len-len(sent)<0 else self.max_len-len(sent)) for sent in sentences]

        # One-hot encode labels
        #X_test_cat_labels = np.array([to_categorical(i, num_classes=self.n_tags) for i in X_test_labels])

        # Generate segment_ids (all 0s) and attention_mask (1 for tokens, 0 for padding)
        segment_ids = np.zeros_like(X_test_token_ids)
        attention_mask = (X_test_token_ids != self.word2idx["<PAD>"]).astype(int)

        dataset=pd.DataFrame()
        dataset['Sentence']=list(dataset)
        dataset["token_id"] = list(X_test_token_ids)
        #dataset["label_id"] = list(X_test_labels)
        dataset["segment_id"] = list(segment_ids)
        dataset["attention_mask"] = list(attention_mask)
        #dataset["categorical_labels"] = list(X_test_cat_labels)
        dataset["padded_sentence"] = list(X_test_padded_sent)

        return dataset
    

def get_predicted_entities(all_prediction, sub_sentence,ignore_labels):

    final_kwords = []
    for i, prediction in enumerate(all_prediction):
        tkns = sub_sentence[i]
        kword = ''
        kword_list = []
        #print(prediction)

        for k, j in enumerate(prediction):
            if (len(prediction)>1):
                if (j not in ignore_labels) & (k==0):
                    # If it's the first word in the first position
                    begin = tkns[k]
                    kword = begin
                elif (j not in ignore_labels) & (k>=1) & (prediction[k-1]==0):
                    # Begin word is in the middle of the sentence
                    begin = tkns[k]
                    previous = tkns[k-1]

                    if begin.startswith('##'):
                        kword = previous + begin[2:]
                    else:
                        kword = begin

                    if k == (len(prediction) - 1):
                        kword_list.append(kword.rstrip().lstrip())
                elif (j not in ignore_labels) & (k>=1) & (prediction[k-1]!=0):
                    # Intermediate word of the same keyword
                    inter = tkns[k]

                    if inter.startswith('##'):
                        kword = kword + "" + inter[2:]
                    else:
                        kword = kword + " " + inter


                    if k == (len(prediction) - 1):
                        kword_list.append(kword.rstrip().lstrip())
                elif (j in ignore_labels) & (k>=1) & (prediction[k-1] !=0):
                    # End of a keywords but not end of sentence.
                    kword_list.append(kword.rstrip().lstrip())
                    kword = ''
                    inter = ''
            else:
                if (j not in ignore_labels):
                    begin = tkns[k]
                    kword = begin
                    kword_list.append(kword.rstrip().lstrip())

        final_kwords.append(kword_list)
    return final_kwords


def visualize_ner(tokens, labels):
    plt.figure(figsize=(24, 2))
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    text_x = 0.05
    text_y = 0.5

    for token, label in zip(tokens, labels):
        color = "lightblue" if label != "O" else "white"
        bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor=color)
        plt.text(text_x, text_y, token, ha="left", va="center", fontsize=14, bbox=bbox_props)

        if label != "O":
            plt.text(text_x, text_y - 0.3, label, ha="left", va="center", fontsize=10, color="red")

        text_x += len(token) * 0.05 + 0.05

    plt.xlim(0, text_x)
    plt.ylim(0, 1)
    plt.axis("off")
    plt.show()



