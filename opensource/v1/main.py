from opensorce_fifa_hackathon_v1 import answer_query_with_context
debug=False
bool=False
import pandas as pd 
df = pd.read_excel('FIFA_2022_Questions.xlsx')
print('no of rows' + str(df.shape[0]))

#df = df.head(10)
df['Answers']=df.Questions.apply(answer_query_with_context);
print(df)
df_filtered = df[['Questions', 'Answers']]


if debug:
    print(df_filtered.columns.tolist())
    print(df_filtered.head())

# Write the DataFrame to an Excel file
df_filtered.to_excel('fifa_openai_v1_output.xlsx', index=False)