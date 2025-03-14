import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import spacy
import networkx as nx
import nltk
import gensim
from gensim import corpora, models
from collections import Counter, defaultdict
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pymysql
import os

# Environment variables with fallbacks
DB_PW = os.getenv("MYSQL_PASSWORD")   # Replace with your actual password
MYSQL_HOST = os.getenv("MYSQL_HOST") #or "database-1.clqcu4ueotm0.us-east-2.rds.amazonaws.com"
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE") #or "BookReviews"
MYSQL_USER = os.getenv("MYSQL_USER") #or "admin"

class DataProcessing:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('en_core_web_sm')
        self.dictionary = None
        self.lda_model = None
        self.df = self.load_data()
        self.df = self.analyze_book_reviews(self.df)
        self.dictionary, self.lda_model, self.df = self.train_lda_model(self.df)
        self.book_topics = self.assign_topics()
        self.G = self.create_graph()
        self.topic_sentences = self.extract_representative_sentences()

    def load_data(self):
        username = MYSQL_USER
        password = DB_PW
        host = MYSQL_HOST
        database = MYSQL_DATABASE
        query = "SELECT * FROM books;"

        try:
            connection = pymysql.connect(
                host=host,
                user=username,
                password=password,
                database=database,
                port=3306,  # Explicitly specify RDS port
                connect_timeout=10
            )
            df = pd.read_sql(query, con=connection)
        except pymysql.Error as err:
            print(f"Database Error: {err}")
            raise
        finally:
            if 'connection' in locals():
                connection.close()

        return df

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    def recognize_entities(self, text):
        doc = self.nlp(text)
        return [(entity.text, entity.label_) for entity in doc.ents]

    def analyze_book_reviews(self, df):
        df['avg_sentiment'] = df['sentiment_polarity'] = df['sentiment_subjectivity'] = 0.0
        df['min_review'] = df['max_review'] = ""
        df['positive_feedback'] = [[] for _ in range(len(df))]
        df['negative_feedback'] = [[] for _ in range(len(df))]
        df['entities'] = [[] for _ in range(len(df))]

        for index, row in df.iterrows():
            title = row['Title']
            reviews = row['merged_reviews'].split('\n')
            sentiments = [TextBlob(review).sentiment for review in reviews]
            polarities = [s.polarity for s in sentiments]

            if polarities:
                df.at[index, 'avg_sentiment'] = np.mean(polarities)
                df.at[index, 'min_review'] = reviews[np.argmin(polarities)]
                df.at[index, 'max_review'] = reviews[np.argmax(polarities)]
                df.at[index, 'positive_feedback'] = [reviews[i] for i, p in enumerate(polarities) if p > 0.1]
                df.at[index, 'negative_feedback'] = [reviews[i] for i, p in enumerate(polarities) if p < -0.1]

            entities = []
            for review in reviews:
                entities.extend(self.recognize_entities(review))
            df.at[index, 'entities'] = entities

        return df

    def create_graph(self):
        G = nx.MultiDiGraph()
        for _, row in self.df.iterrows():
            G.add_node(row['Title'], type='Book', sentiment=row['avg_sentiment'])
            G.add_node(row['authors'], type='Author')
            G.add_edge(row['Title'], row['authors'], relation='WRITTEN_BY')
            G.add_node(row['publisher'], type='Publisher')
            G.add_edge(row['Title'], row['publisher'], relation='PUBLISHED_BY')
        return G

    def train_lda_model(self, df):
        df['tokenized_review'] = df['merged_reviews'].fillna("").apply(
            lambda x: [word for word in word_tokenize(x) if word not in self.stop_words])
        dictionary = corpora.Dictionary(df['tokenized_review'])
        corpus = [dictionary.doc2bow(text) for text in df['tokenized_review']]
        lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=7, passes=15)
        df['topics'] = df['tokenized_review'].apply(
            lambda x: sorted(lda_model[dictionary.doc2bow(x)], key=lambda tup: -tup[1])[0][0] if x else None)
        return dictionary, lda_model, df

    def assign_topics(self):
        return self.df.groupby('Title')['topics'].apply(lambda x: x.value_counts().idxmax()).reset_index()

    def extract_representative_sentences(self):
        topic_sentences = defaultdict(list)
        for _, row in self.df.iterrows():
            bow = self.dictionary.doc2bow(row['tokenized_review'])
            topics = self.lda_model.get_document_topics(bow)
            if topics:
                dominant_topic = sorted(topics, key=lambda tup: -tup[1])[0][0]
                sentences = sent_tokenize(row['merged_reviews'])[:3]
                topic_sentences[dominant_topic].extend(sentences)
        return topic_sentences

    def get_common_entities(self, title, entity_type=None):
        book_reviews = self.df[self.df['Title'] == title]
        entities = [entity for sublist in book_reviews['entities'] for entity in sublist]
        if entity_type:
            entities = [entity for entity in entities if entity[1] == entity_type]
        return Counter(entities).most_common()

DP = DataProcessing()
df, G, dictionary, lda_model, book_topics, topic_sentences = DP.df, DP.G, DP.dictionary, DP.lda_model, DP.book_topics, DP.topic_sentences

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import re
# import spacy
# import networkx as nx
# import nltk
# import gensim
# from gensim import corpora, models
# from collections import Counter, defaultdict
# from sqlalchemy import create_engine
# from textblob import TextBlob
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import pymysql
# import os

# DB_PW = os.getenv("MYSQL_PASSWORD")
# MYSQL_HOST = os.getenv("MYSQL_HOST")
# MYSQL_DATABASE= os.getenv("MYSQL_DATABASE")
# MYSQL_USER= os.getenv("MYSQL_USER")


# class DataProcessing:
#     def __init__(self):
#         self.stop_words = set(stopwords.words('english'))
#         self.lemmatizer = WordNetLemmatizer()
#         self.nlp = spacy.load('en_core_web_sm')
#         self.dictionary = None
#         self.lda_model = None
#         self.df = self.load_data()
#         self.df = self.analyze_book_reviews(self.df)
#         self.dictionary, self.lda_model, self.df = self.train_lda_model(self.df)
#         self.book_topics = self.assign_topics()
#         self.G = self.create_graph()
        
#         self.topic_sentences = self.extract_representative_sentences()
    
#     import pymysql

#     def load_data(self):
#         username = MYSQL_USER
#         password = DB_PW 
#         host = MYSQL_HOST
#         database = MYSQL_DATABASE
#         query = "SELECT * FROM books;"

#         connection = pymysql.connect(
#             host=host,
#             user=username,
#             password=password,
#             database=database
#         )

#         try:
            
#             df = pd.read_sql(query, con=connection)
#         finally:
#             connection.close()  #

#         return df

#     def preprocess_text(self, text):
#         text = text.lower()
#         text = re.sub(r'\d+', '', text)
#         text = re.sub(r'[^\w\s]', '', text)
#         text = re.sub(r'\s+', ' ', text)
#         words = text.split()
#         words = [word for word in words if word not in self.stop_words]
#         words = [self.lemmatizer.lemmatize(word) for word in words]
#         return ' '.join(words)
    
#     def recognize_entities(self, text):
#         doc = self.nlp(text)
#         return [(entity.text, entity.label_) for entity in doc.ents]
    
   
#     def analyze_book_reviews(self, df):
#         df['avg_sentiment'] = df['sentiment_polarity'] = df['sentiment_subjectivity'] = 0.0
#         df['min_review'] = df['max_review'] = ""
#         df['positive_feedback'] = [[] for _ in range(len(df))]
#         df['negative_feedback'] = [[] for _ in range(len(df))]
#         df['entities'] = [[] for _ in range(len(df))]  # Add entities column
        
#         for index, row in df.iterrows():
#             title = row['Title']
#             reviews = row['merged_reviews'].split('\n')
#             sentiments = [TextBlob(review).sentiment for review in reviews]
#             polarities = [s.polarity for s in sentiments]
            
#             if polarities:
#                 df.at[index, 'avg_sentiment'] = np.mean(polarities)
#                 df.at[index, 'min_review'] = reviews[np.argmin(polarities)]
#                 df.at[index, 'max_review'] = reviews[np.argmax(polarities)]
#                 df.at[index, 'positive_feedback'] = [reviews[i] for i, p in enumerate(polarities) if p > 0.1]
#                 df.at[index, 'negative_feedback'] = [reviews[i] for i, p in enumerate(polarities) if p < -0.1]
            
           
#             entities = []
#             for review in reviews:
#                 entities.extend(self.recognize_entities(review))  
#             df.at[index, 'entities'] = entities  
        
#         return df

#     def create_graph(self):
#         G = nx.MultiDiGraph()
#         for _, row in self.df.iterrows():
        
#             G.add_node(row['Title'], type='Book', sentiment=row['avg_sentiment'])
            
#             G.add_node(row['authors'], type='Author')  
#             G.add_edge(row['Title'], row['authors'], relation='WRITTEN_BY')
            
            
#             G.add_node(row['publisher'], type='Publisher') 
#             G.add_edge(row['Title'], row['publisher'], relation='PUBLISHED_BY')
            
#         return G

#     def train_lda_model(self, df):
#         df['tokenized_review'] = df['merged_reviews'].fillna("").apply(lambda x: [word for word in word_tokenize(x) if word not in self.stop_words])
#         dictionary = corpora.Dictionary(df['tokenized_review'])
#         corpus = [dictionary.doc2bow(text) for text in df['tokenized_review']]
#         lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=7, passes=15)
#         df['topics'] = df['tokenized_review'].apply(lambda x: sorted(lda_model[dictionary.doc2bow(x)], key=lambda tup: -tup[1])[0][0] if x else None)
#         return dictionary, lda_model, df

#         #return dictionary, lda_model
    
#     def assign_topics(self):
#         return self.df.groupby('Title')['topics'].apply(lambda x: x.value_counts().idxmax()).reset_index()
    
#     def extract_representative_sentences(self):
#         topic_sentences = defaultdict(list)
#         for _, row in self.df.iterrows():
#             bow = self.dictionary.doc2bow(row['tokenized_review'])
#             topics = self.lda_model.get_document_topics(bow)
#             if topics:
#                 dominant_topic = sorted(topics, key=lambda tup: -tup[1])[0][0]
#                 sentences = sent_tokenize(row['merged_reviews'])[:3]
#                 topic_sentences[dominant_topic].extend(sentences)
#         return topic_sentences
    
#     def get_common_entities(self, title, entity_type=None):
#         book_reviews = self.df[self.df['Title'] == title]
#         entities = [entity for sublist in book_reviews['entities'] for entity in sublist]
#         if entity_type:
#             entities = [entity for entity in entities if entity[1] == entity_type]
#         return Counter(entities).most_common()
        
      
# DP = DataProcessing()
# df, G, dictionary, lda_model, book_topics, topic_sentences = DP.df, DP.G, DP.dictionary, DP.lda_model, DP.book_topics, DP.topic_sentences


