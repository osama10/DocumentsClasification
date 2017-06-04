#Loading the datafile
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

#Extracting features in form of bag of words
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

#Using tf-id
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#trainig classifier

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

#prediction
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

#using pipeline
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

#performance evaluation
import numpy as np
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
mean_from_nb = np.mean(predicted == twenty_test.target)
print(mean_from_nb)


#using svm
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42)),])
_ = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
mean_from_svm = np.mean(predicted == twenty_test.target)
print(mean_from_svm)

# detailed performance analysis
from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))
