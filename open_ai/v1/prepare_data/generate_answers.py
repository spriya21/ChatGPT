
import pandas as pd
import tiktoken
import openai
openai.api_key =  "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"
from openai.embeddings_utils import distances_from_embeddings
import pandas as pd
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity


df=pd.read_csv('embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

df.head()

debug = True
################################################################################
### Step 12
################################################################################

def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=True,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the questin and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )

        print("question:" + question)
        print("answer:" + response["choices"][0]["text"].strip())
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

# ################################################################################
# ### Step 13
# ################################################################################

import pandas as pd
questions = pd.read_excel('FIFA_2022_Questions_v2.xlsx')
questions = questions[questions['is_valid'] == 'Y']

# questions = questions.head(5)
print('no of rows' + str(questions.shape[0]))


#df = df.head(10)
questions['generated_response']=questions.Questions.apply(lambda x: answer_question(df, question=x));
df_filtered = questions[['Questions', 'generated_response']]

if debug:
    print(df_filtered.columns.tolist())
    print(df_filtered.head())

# Write the DataFrame to an Excel file
df_filtered.to_excel('fifa_openai_v3_output.xlsx', index=False)
questions.to_excel('fifa_openai_v3_raw_output.xlsx', index=False)