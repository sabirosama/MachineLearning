from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer() ##for comparing sentences and find similarity

tfidf = vectorizer.fit_transform(['I like Machine Learning and clustering algorithms',
                                  'Apples, oranges and any kind of fruits are healthy',
                                  'Is it feasible with machine learning algorithm?',
                                  'My family is happy because of the healthy fruits'])
#print(tfidf.A)
print((tfidf*tfidf.T).A) ##relation between sentences in matrix