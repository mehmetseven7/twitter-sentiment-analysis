import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#read csv file
df = pd.read_csv('Train_this.csv')

#Divide the dataset for train and test
X_train, X_test, y_train, y_test = train_test_split(df['Lemmatized_Tweets'], df['Label'], test_size=0.2, random_state=42)

#Convert texts to vector
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

#Create and train Naive Bayes Model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

#Prediction on Test Set
y_pred = nb_classifier.predict(X_test_vectorized)

#review accuracy precision recall and f1 score
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")