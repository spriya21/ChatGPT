from urllib.parse import urlparse

# Open the file containing the URLs
with open('urls.txt', 'r') as f:
    # Read the contents of the file into a list of URLs
    urls = f.readlines()

# Initialize a set to store unique URLs
unique_urls = set()

# Loop through each URL in the list
for url in urls:
    # Remove any leading or trailing whitespace from the URL
    url = url.strip()
    parsed_url = urlparse(url)
    main_url = parsed_url.scheme + '://' + parsed_url.netloc + parsed_url.path
    
    # Add the URL to the set of unique URLs
    unique_urls.add(main_url)

# Convert the set of unique URLs back to a list
unique_urls = list(unique_urls)

# Open the file in write mode
with open('urls_cleaned.txt', 'w') as f:
    # Loop through the unique URLs and write them to the file on a new line
    for url in unique_urls:
        f.write(url + '\n')