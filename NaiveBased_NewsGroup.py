from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']

##the subset is train,categories are above mentioned, shuffle is on, and random state
training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

##we just count the word occurrences
count_vector = CountVectorizer()
x_train_counts = count_vector.fit_transform(training_data.data)


tfid_transformer = TfidfVectorizer()
x_train_tfidf = tfid_transformer.fit_transform(x_train_counts)

model = MultinomialNB().fit(x_train_tfidf, training_data.targets)

new = ['This is nothing to do with the religion or church', 'Software engineering is getting hotter and hotter now a days']

##convert new into numerical values

x_new_counts = count_vector.transform(new)
x_new_tfidf = tfid_transformer.transform(x_new_counts)

predicted = model.predict(x_new_tfidf)

#print(predicted)

##for loop for the sentence=doc,
for doc, category in zip(new, predicted):
    print('%r ------> %s' %(doc, training_data.target_names[category]))
