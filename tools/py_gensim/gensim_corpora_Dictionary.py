from gensim.corpora import Dictionary


#
# doc2bow，将document转换成bow(bag of words)，(token_id, token_counts)
# 1.documents
texts = [['humans', 'interface', 'computer'], ["hello", "wife", "computer"]]

# 2. create a Dictionary
dct = Dictionary(texts)

# 3. add new documents
dct.add_documents([["cat", "say", "meow", "dog"], ["dog"]])

# 4. watch Dictionary items
for key, value in dct.iteritems():
    print(key, value)

# 5. doc to bag of words
bow = dct.doc2bow(["dog", "computer", "non_existent_word", "computer"])
print(bow)

# 6. filter out extremes
dct.filter_extremes(no_below=1, no_above=0.5, keep_n=3)
for key, value in dct.iteritems():
    print(key, value)

from gensim import corpora, models

tfidf = models.TfidfModel(bow)
corpus_tfidf = tfidf[bow]

from Z
