### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 02.09.2026
### NAME: KARTHIKEYAN D
### Register No:212224230115


### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:
```

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?"
}

def preprocess(text):
    return " ".join(x for x in word_tokenize(text.lower())
                    if x not in stopwords.words("english")
                    and x not in string.punctuation)

docs = [preprocess(x) for x in documents.values()]

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(docs)

query = preprocess(input("Enter query: "))
query_vector = vectorizer.transform([query])

scores = cosine_similarity(query_vector, matrix)[0]

for rank, i in enumerate(scores.argsort()[::-1], 1):
    print("----------------------")
    print("Rank:", rank)
    print("Document:", list(documents)[i])
    print("Similarity Score:", scores[i])


```


### Output:

<img width="503" height="410" alt="image" src="https://github.com/user-attachments/assets/b37a6f2d-c969-470b-808b-c7d3edda5646" />

### Result:
Thus, the implementation of Information Retrieval Using Vector Space Model in Python is executed successfully.
