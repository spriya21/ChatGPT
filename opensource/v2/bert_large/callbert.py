from bert_large_cased import answer_query_with_context
# from gpt2 import answer_query_with_context
# from embed import answer_query_with_context
debug=False
bool=False
import pandas as pd 
df = pd.read_excel('fifa_openai_v2_raw_output_verified.xlsx')
print('no of rows' + str(df.shape[0]))

#df = df.head(10)
df['Answers']=df.Questions.apply(answer_query_with_context);
print(df)
df_filtered = df[['Questions', 'Answers']]


if debug:
    print(df_filtered.columns.tolist())
    print(df_filtered.head())

# Write the DataFrame to an Excel file
df_filtered.to_excel('Answers.xlsx', index=False)