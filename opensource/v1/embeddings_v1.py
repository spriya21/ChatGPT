# -*- coding: utf-8 -*-
"""
opensource version1 Eureka Incite Hackathon

Prerequesites:
pip install -r requirements.txt

"""

import torch
import pandas as pd
import requests
from retry import retry

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import BertTokenizer, BertForQuestionAnswering
from sentence_transformers.util import semantic_search
from sentence_transformers import SentenceTransformer

def get_content(path):
  df = pd.read_csv(path)
  out = ' '.join(df["text"])
  texts=[]
  while out:
    # Add the first 256 characters to the grouping
    texts.append(out[:1024])
    # Set the contents to everything after the first 256
    out = out[1024:]
  return texts

@retry(tries=3, delay=10)
def query(texts):
    # model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model_id="sentence-transformers/all-mpnet-base-v2"
    hf_token = "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"
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.post(api_url, headers=headers, json={"inputs": texts})
    result = response.json()
    if isinstance(result, list):
      return result
    elif list(result.keys())[0] == "error":
      raise RuntimeError(
          "The model is currently loading, please re-run the query."
          )

def get_top_k_articles(question,docs,k=2):
    model_id = "sentence-transformers/all-mpnet-base-v2"
    hf_token = "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"
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    texts=get_content('scraped.csv')
    output = query(texts)
    dataset_embeddings = torch.FloatTensor(output)
    query_output = query(question)
    query_embeddings = torch.FloatTensor(query_output)
    hits = semantic_search(query_embeddings, dataset_embeddings, top_k=1)
    return [texts[hits[0][i]['corpus_id']] for i in range(len(hits[0]))]



def answer_question(question, answer_text):

    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    input_ids = tokenizer.encode(question, answer_text, max_length=512)
    
    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    outputs = model(torch.tensor([input_ids]), # The tokens representing our input text.
                    token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                    return_dict=True) 

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    print('Answer: "' + answer + '"')
    print("_________________________________________")
    print("")
    return answer


def answer_query_with_context(query):
  # Get the list of the content
  candidate_docs = get_top_k_articles(query, 1)

  # Print the answer
  answer=answer_question(query,candidate_docs[0])
  return answer