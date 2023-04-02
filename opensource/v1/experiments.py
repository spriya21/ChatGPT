# -*- coding: utf-8 -*-
"""
opensource version1 Eureka Incite Hackathon

Prerequesites:
pip install -r requirements.txt

"""

import torch
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline
def get_content(path):
  df = pd.read_csv(path)
  out = ' '.join(df["text"])
  col_list = out.split("\n")
  return col_list

def segment_documents(docs, max_doc_length=450):
  # List containing full and segmented docs
  segmented_docs = []

  for doc in docs:
    # Split document by spaces to obtain a word count that roughly approximates the token count
    split_to_words = doc.split(" ")

    # If the document is longer than our maximum length, split it up into smaller segments and add them to the list 
    if len(split_to_words) > max_doc_length:
      for doc_segment in range(0, len(split_to_words), max_doc_length):
        segmented_docs.append( " ".join(split_to_words[doc_segment:doc_segment + max_doc_length]))

    # If the document is shorter than our maximum length, add it to the list
    else:
      segmented_docs.append(doc)

  return segmented_docs

def get_top_k_articles(query, docs, k=2):

  # Initialize a vectorizer that removes English stop words
  vectorizer = TfidfVectorizer(analyzer="word", stop_words='english')

  # Create a corpus of query and documents and convert to TFIDF vectors
  query_and_docs = [query] + docs
  matrix = vectorizer.fit_transform(query_and_docs)

  # Holds our cosine similarity scores
  scores = []

  # The first vector is our query text, so compute the similarity of our query against all document vectors
  for i in range(1, len(query_and_docs)):
    scores.append(cosine_similarity(matrix[0], matrix[i])[0][0])

  # Sort list of scores and return the top k highest scoring documents
  sorted_list = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
  top_doc_indices = [x[0] for x in sorted_list[:k]]
  top_docs = [docs[x] for x in top_doc_indices]
  
  return top_docs

def answer_query_with_context(query):
  # Get the list of the content
  col_list = get_content("scraped.csv")

  # Segment our documents
  segmented_docs = segment_documents(col_list, 200)

  # Retrieve the top k most relevant documents to the query
  candidate_docs = get_top_k_articles(query, segmented_docs, 1)

  # Print the answer
  model_name='distilbert-base-uncased-distilled-squad'
  # model_name = "deepset/roberta-base-squad2"
  # model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
  # model_name = "deepset/minilm-uncased-squad2"
  # model_name = "deepset/roberta-base-squad2-covid"
  question_answerer = pipeline("question-answering",model=model_name)
  result = question_answerer(query,candidate_docs[0])
  answer = result['answer']
  #answer=answer_question(query,candidate_docs[0])
  return answer

