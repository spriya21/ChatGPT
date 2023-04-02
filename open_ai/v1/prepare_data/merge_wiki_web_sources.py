import pandas as pd 

wiki_df = pd.read_csv('./wiki/fifa_wiki_sections_merged.csv')
wiki_df = wiki_df[['title', 'heading', 'content']]
wiki_df = wiki_df.assign(title=wiki_df['title'] + '_' + wiki_df['heading'])
wiki_df = wiki_df[['title', 'content']]
wiki_df = wiki_df.rename(columns={'content': 'text'})
################################################################################
### Step 7
################################################################################

df = pd.read_csv('./web/scraped_context.csv', index_col=0)
df.columns = ['title', 'text']
print(df.shape[0])
df = pd.concat([df, wiki_df])
print(df.shape[0])

df.to_csv('fifa_wiki_web_data.csv', index=False)
df.to_excel('fifa_wiki_web_data.xlsx', index=False)