# Named Entity Recognition (NER) for Twitter Data

## Problem Statement
Twitter is a microblogging and social networking platform where millions of tweets are generated daily. To analyze trends and extract meaningful insights, Twitter aims to develop an automated system to identify named entities in tweets without relying on user-defined hashtags. This system will recognize various entity types, including persons, locations, companies, and other named categories, despite inconsistencies in user-generated text.

## Insights from the Dataset

### 1. Class Imbalance in Named Entity Labels
The dataset exhibits severe class imbalance, which can negatively impact model performance. The vast majority of tokens are labeled as `O` (outside any named entity), making up ~93% of the dataset (44,803 out of 48,298 total labels).

The top three entity types (excluding `O`) are:
- **B-person** (0.93%)
- **B-geo-loc** (0.57%)
- **B-other** (0.47%)

Some labels have very few occurrences, such as `I-sportsteam` (0.048%), making them difficult for a model to learn effectively.

### 2. Small Dataset Size
- The dataset contains only **2,394 sentences** (test set has **3,850 sentences**), with an average sentence length of **19.74 words**.
- The shortest tweet has just **1 word**, while the longest has **41 words**.
- A **larger dataset** would be necessary for a model to generalize well, especially for low-frequency entities.

### 3. Named Entity Distribution in Sentences

| Named Entity | % of Sentences Containing It | % of Words Assigned to It |
|-------------|-----------------------------|--------------------------|
| B-person    | 14.41%                      | 0.95%                    |
| B-geo-loc   | 8.90%                       | 0.58%                    |
| B-other     | 8.06%                       | 0.48%                    |
| I-person    | 6.93%                       | 0.45%                    |
| B-company   | 6.31%                       | 0.36%                    |
| I-other     | 5.85%                       | 0.68%                    |
| B-facility  | 3.93%                       | 0.22%                    |

- `B-person` appears in the most sentences (14.41%) but still accounts for less than 1% of total words.
- Rare entities (`I-sportsteam`, `I-tvshow`, `I-movie`) have an extremely low presence, making classification difficult.

## Recommendations

### 1. Address Class Imbalance
- Use **data augmentation techniques** (e.g., synonym replacement, back-translation) to artificially increase the size of underrepresented categories.
- Apply **weighted loss functions** or **focal loss** to give more importance to underrepresented labels.
- Consider using **sampling techniques**:
  - **Oversampling** for minority classes.
  - **Undersampling** for the `O` class to prevent the model from learning a bias toward non-entities.

### 2. Expand the Dataset
- The current dataset is **too small** for robust deep learning models. Consider:
  - Collecting **more labeled Twitter data**.
  - Using **pre-trained NER models** (e.g., spaCy, BERT-based NER) for weak supervision on unlabeled data.

### 3. Model Considerations
- **Baseline Model:** A **BiLSTM-CRF** model is a good starting point but might struggle with rare entities.
- **Advanced Model:** Use a **pre-trained transformer-based NER model** (e.g., BERT, RoBERTa, T5) to leverage contextual word representations.
- **Next step:** A **hybrid approach** combining **rule-based methods** (e.g., regex for common names) with deep learning to improve rare class recognition.

### 4. Improve Label Consistency
- Merge **low-frequency labels** (e.g., `B-movie`, `B-tvshow`, `B-musicartist` → `B-entertainment`) to create more balanced classes.
- Standardize **label definitions** to avoid over-segmentation of entities.
- Here, all entities have been merged into a single category: **'E'**.

### 5. Use Data Preprocessing for Better Recognition
- **Lowercasing & spell correction** (e.g., `wordninja`, `pyspellchecker`) were used to fix user typos.
- **Hashtag and emoji expansion** can be done to preserve meaningful entities.

By implementing these recommendations, the NER model can achieve **better generalization and performance** despite dataset limitations.

---

## Data Preprocessing & Tokenization

1. Lowercased text and expanded contractions.
2. Implemented a **Custom Tokenizer** (Word2Vec-based) and a **BERT Tokenizer**:
   - **Custom Tokenizer:** Spell-checking, word segmentation, special token replacement, and pre-trained Word2Vec embeddings.
   - **BERT Tokenizer:** Subword tokenization while preserving entity labels.

## Dataset Preparation

1. Processed tokenized text into lists of **token IDs, attention masks, segment IDs, and label IDs** (both one-hot and label-coded).
2. Padded sequences and split the dataset into **train & validation** sets.
3. Calculated **class weights** to balance the model, adjusting for rare entities and reducing weight for `O` labels.

## Model Training & Performance

Two models were trained:

1. **BiLSTM+CRF:** Leveraged a custom tokenizer and CRF-based sequence modeling.
2. **BERT-based model:** Fine-tuned `bert-base-uncased` for token classification.

### Model Results
- The **BiLSTM+CRF model** failed to recognize entities, predicting every token as `O`, despite achieving **high overall accuracy**. This suggests that while it handled majority class tokens well, it struggled with **rare entity labels**. The use of **Sigmoid Focal Loss** with class weighting **did not improve entity recognition**.
- In contrast, the **BERT model** performed significantly better, achieving **higher accuracy** while still facing challenges in entity classification. Although it improved upon the custom model, it **still struggled** to correctly classify **non-'O' labels**, highlighting the difficulty in learning entity boundaries with limited data.

## Key Takeaways
- **BiLSTM+CRF failed** to capture entities, likely due to limitations in feature extraction or ineffective class weighting.
- **BERT outperformed BiLSTM+CRF**, but still had difficulty with **rare entity labels**, suggesting that more data or better **entity-specific augmentation** could improve results.
- **Future improvements** could involve **increasing data size**, fine-tuning hyperparameters, and trying **more sophisticated regularization techniques**.

## Deployment
A **Streamlit app** was created using the **BERT model**, making the trained model accessible for real-world use.
