# %%
# !pip install beautifulsoup4
# !pip install numpy
# !pip install requests
# !pip install spacy
# !pip install trafilatura

# %%
from bs4 import BeautifulSoup
import json
import numpy as np
import requests
from requests.models import MissingSchema
import spacy
import trafilatura

# %%
testsite_array = []
with open('urls_cleaned.txt') as my_file:
    for line in my_file:
        testsite_array.append(line)

testsite_array=[i.replace("\n","") for i in testsite_array]

def beautifulsoup_extract_text_fallback(response_content):
    
    '''
    This is a fallback function, so that we can always return a value for text content.
    Even for when both Trafilatura and BeautifulSoup are unable to extract the text from a 
    single URL.
    '''
    
    # Create the beautifulsoup object:
    soup = BeautifulSoup(response_content, 'html.parser')
    
    # Finding the text:
    text = soup.find_all(text=True)
    
    # Remove unwanted tag elements:
    cleaned_text = ''
    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head', 
        'input',
        'script',
        'style',]

    # Then we will loop over every item in the extract text and make sure that the beautifulsoup4 tag
    # is NOT in the blacklist
    for item in text:
        if item.parent.name not in blacklist:
            cleaned_text += '{} '.format(item)
            
    # Remove any tab separation and strip the text:
    cleaned_text = cleaned_text.replace('\t', '')
    return cleaned_text.strip()
    

def extract_text_from_single_web_page(url):
    print('scrapping..' + url)
    downloaded_url = trafilatura.fetch_url(url)
    try:
        a = trafilatura.extract(downloaded_url, output_format="json", with_metadata=True, include_comments = False,
                            date_extraction_params={'extensive_search': True, 'original_date': True})
    except AttributeError:
        a = trafilatura.extract(downloaded_url, output_format="json", with_metadata=True,
                            date_extraction_params={'extensive_search': True, 'original_date': True})
    if a:
        json_output = json.loads(a)
        return json_output['text']
    else:
        try:
            resp = requests.get(url)
            # We will only extract the text from successful requests:
            if resp.status_code == 200:
                return beautifulsoup_extract_text_fallback(resp.content)
            else:
                # This line will handle for any failures in both the Trafilature and BeautifulSoup4 functions:
                return np.nan
        # Handling for any URLs that don't have the correct protocol
        except MissingSchema:
            return np.nan

    
dictionary={}
for single_url in testsite_array:
    try:
        text = extract_text_from_single_web_page(url=single_url)
        dictionary[single_url]=text
    except Exception:
        pass

import pandas as pd

df = pd.DataFrame(list(dictionary.items()),columns = ['links','content']) 


# df.to_csv('scraped_context.csv')
df.to_csv('scraped_context.csv', escapechar='\\')

# df.to_excel('scraped_context.xlsx')

# define a function to remove illegal characters
import re
def remove_illegal_chars(cell_value):
    if cell_value is not None:
        # remove null characters and line breaks
        cell_value = re.sub('[\0\n\r\t]', '', cell_value)
    return cell_value

# apply the function to each cell in the DataFrame
df = df.applymap(remove_illegal_chars)
df.to_excel('scraped_context.xlsx')

