from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from math import log as log
import csv
import string
import numpy as np

corpus = [] 
heading_corpus = [] 
docs_to_terms = {} 
docs_to_heading_terms = {}
terms_to_docs = dict()
doc_length_normalisation = {}

i = 0
total_words = 0
with open(r'dataSet.csv', 'r', newline='') as myfile:
    wr = csv.reader(myfile, quoting=csv.QUOTE_ALL)
    for row in wr:
        heading_corpus.append(row[2])
        corpus.append(row[5])
        total_words += len(row[5])
        i += 1
       

N = len(corpus)


def tokenization(doc, doc_id, flag):
    tokens = [w for w in word_tokenize(doc.casefold()) if not w in set(stopwords.words("english") + list(string.punctuation))]
    stemmed_tokens = sorted([PorterStemmer().stem(w) for w in tokens])
    temp_lst = dict()   
    
    if flag == 0:
        for w in stemmed_tokens:
            if w in terms_to_docs:
                if doc_id not in terms_to_docs[w]:
                    terms_to_docs[w].append(doc_id)
            else:
                terms_to_docs[w] = [doc_id]
        
            temp_lst[w] = stemmed_tokens.count(w)
        return temp_lst
    
    else:
        for w in stemmed_tokens:
            if w in docs_to_heading_terms:
                if doc_id not in docs_to_heading_terms[w]:
                    docs_to_heading_terms[w].append(doc_id)
            else:
                docs_to_heading_terms[w] = [doc_id]
        
        return None


i = 0
for doc in corpus:
    docs_to_terms[i] = tokenization(doc, i, 0)
    i += 1
    
i = 0
for doc in heading_corpus:
    tokenization(doc, i, 1)
    i += 1

for doc in docs_to_terms:
    summ = 0
    for word in docs_to_terms[doc]:
        tf = log(1+(docs_to_terms[doc][word]), 10)
        idf = log((N/len(terms_to_docs[word])), 10)
        docs_to_terms[doc][word] = tf * idf
        summ += (docs_to_terms[doc][word]) ** 2
    doc_length_normalisation[doc] = summ**0.5

for doc in docs_to_terms:
    for word in docs_to_terms[doc]:
        temp = docs_to_terms[doc][word]
        docs_to_terms[doc][word] = temp/doc_length_normalisation[doc]


np.save('docs_to_terms.npy', docs_to_terms) 
np.save('terms_to_docs.npy', terms_to_docs) 
np.save('docs_to_heading_terms.npy', docs_to_heading_terms) 

print("Total words in the corpus", total_words)
print("Number of words per document", total_words/N)