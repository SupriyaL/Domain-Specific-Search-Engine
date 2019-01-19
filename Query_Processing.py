from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import heapq
import time
import string
import numpy as np
import csv
from tkinter import *

QUERY = ""


def searchResult(x):
    label = Label(top, text=x)
    label.pack()


corpus = {}
i = 0

with open('dataSet.csv', 'r', newline='') as myfile:
    wr = csv.reader(myfile, quoting=csv.QUOTE_ALL)
    for row in wr:
        corpus[i] = str(row[5])
        i += 1


def tokenization(doc, doc_id, flag=0):
    tokens = [w for w in word_tokenize(doc.casefold()) if
              not w in set(stopwords.words("english") + list(string.punctuation))]
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


terms_to_docs = np.load('terms_to_docs.npy', encoding='bytes').item()
docs_to_terms = np.load('docs_to_terms.npy', encoding='bytes').item()
docs_to_heading_terms = np.load('docs_to_heading_terms.npy', encoding='bytes').item()

top = Tk()
top.wm_title("Vector Space Model")
intro = Label(top, text="Domain Specific Search Engine", bg="grey", fg="black")
intro.pack(fill=X)
welcome = Label(top, text="Welcome")
welcome.pack(fill=X)
L1 = Label(top, text="Query: ")
L1.pack(side=LEFT, anchor=NW)
E1 = Entry(top, bd=5, width=50)
E1.pack(side=LEFT, anchor=NW, fill=X)


def on_button_click():
    global QUERY
    NO_WORD = 0
    related_docs = {}
    summ = 0
    QUERY = (E1.get())
    if len(QUERY) == 0:
        searchResult("Query Cannot Be Empty")
        QUERY = (E1.get())
    else:
        start = time.time()
        query_terms = tokenization(QUERY, 0)

        if len([i for i in query_terms if i in terms_to_docs]) < 1:
            searchResult("Sorry! Could not find any terms related to the query in the Corpus")
            searchResult("--------------------------------------------------------")
        else:
            for word in query_terms:
                summ += (query_terms[word]) ** 2
                query_length_normalisation = summ ** 0.5

            for word in query_terms:
                temp = query_terms[word]
                query_terms[word] = temp / query_length_normalisation
                for doc_id in terms_to_docs[word]:
                    if doc_id not in related_docs:
                        related_docs[doc_id] = 1
                    else:
                        related_docs[doc_id] += 1

            for doc in related_docs:
                summ = 0
                for word in query_terms:
                    if word in docs_to_terms[doc]:
                        summ += (docs_to_terms[doc][word] * query_terms[word])

                    if word in docs_to_heading_terms:
                        l = docs_to_heading_terms[word]
                        for x in l:
                            if x in related_docs:
                                summ += 1

                related_docs[doc] = summ

            related_docs = heapq.nlargest(10, related_docs, key=related_docs.get)
        if len(related_docs) == 1:
            text.delete('1.0', END)
            text.insert(INSERT, "Sorry! Could not find any documents related to the query in the Corpus")
        else:
            i = 1
            text.delete('1.0', END)
            for doc in related_docs:
                text.insert(INSERT,
                            str(i) + '.' + " " + '(' + "Doc No." + str(doc + 1) + ')' + " " + str(corpus[doc]) + '\n\n')
                text.pack(side=LEFT, fill=Y)
                i += 1
        print("TIME TAKEN TO PROCESS", time.time() - start, "seconds")


button1 = Button(top, text='Search', command=on_button_click)
button2 = Button(top, text='Quit', command=top.quit)
button1.pack(side=LEFT, anchor=NW)
button2.pack(side=LEFT, anchor=NW, fill=X)
resultsFrame = Frame(top)
resultsFrame.pack(side=LEFT, anchor=W)
text = Text(resultsFrame, height=500, width=600, bd=5)
S = Scrollbar(resultsFrame, command=text.yview)
S.pack(side=RIGHT, fill=Y)
text.config(yscrollcommand=S.set)
top.mainloop()
