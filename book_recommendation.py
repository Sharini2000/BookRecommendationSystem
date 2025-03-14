import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
from DataProcessing import DP  

df, G, dictionary, lda_model, book_topics, topic_sentences = DP.df, DP.G, DP.dictionary, DP.lda_model, DP.book_topics, DP.topic_sentences

# Load summarization model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def summarization(text):
    if isinstance(text, list):
        text = ' '.join(text)
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def bookInformation(title):
    if title not in G:
        return f"Sorry, we might not have the book you are looking for. Some of the books which details we have are afterworld, bell toll, love dog, magician tale, vampire story and more."
    feedback = f"Feedback for '{title}':\n"
    sentiment = df.loc[df['Title'] == title, 'avg_sentiment'].values[0]
    feedback += f"Overall sentiment: {'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'}\n"
    author = [n for n in G.neighbors(title) if G.nodes[n]['type'] == 'Author']
    if author:
        feedback += f"The book is written by '{author[0]}'.\n"
    return feedback

def generate_feedback(title, sentiment_type):
    if title not in G:
        return f"No feedback available for '{title}'or we might not have the book you are looking for. Some of the books which details we have are afterworld, bell toll, love dog, magician tale, vampire story and more."
    feedback = f"Feedback for '{title}':\n"
    key = 'positive_feedback' if sentiment_type == 'positive' else 'negative_feedback'
    summary = summarization(df.loc[df['Title'] == title, key].values[0])
    review = df.loc[df['Title'] == title, 'max_review' if sentiment_type == 'positive' else 'min_review'].values[0]
    feedback += f"Summary of {sentiment_type} review: \"{summary}\"\n\nMost {sentiment_type} review: \"{review}\"\n"
    return feedback


def get_recommendations_with_entities(book_title, top_n=5):
    try:
        
        recommendations = DP.get_common_entities(book_title)
        
        # Check if recommendations is a valid DataFrame and contains the 'Title' column
        if not isinstance(recommendations, pd.DataFrame):
            raise ValueError("Recommendations should be a pandas DataFrame.")
        
        if 'Title' not in recommendations.columns:
            raise ValueError("The DataFrame does not contain a 'Title' column.")
        
       
        recommended_books = recommendations['Title'].unique()
        
       
        if isinstance(recommended_books, (np.ndarray, list)):
            return recommended_books[:top_n] 
        else:
            
            return list(recommended_books)[:top_n]
    except Exception as e:
        return ["An error occurred, please try later!"]

def provide_concise_feedback(book_title, num_sentences=3):
    if book_title not in book_topics['Title'].values:
        return f"No topics found for '{book_title}'."
    main_topic = book_topics[book_topics['Title'] == book_title]['topics'].values[0]
    topic_words = lda_model.show_topic(main_topic, topn=7)
    feedback = f"Main topics for '{book_title}': {[word for word, _ in topic_words]}\n\nKey points from reviews:\n"
    if main_topic not in topic_sentences:
        return f"No representative sentences found for '{book_title}'."
    summarized = summarization(" ".join(topic_sentences[main_topic]))
    feedback += summarized
    return feedback

