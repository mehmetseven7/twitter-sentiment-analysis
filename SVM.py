import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics

# CSV dosyanızı okuyun
df = pd.read_csv('Train_this.csv')

# Veri kümesini eğitim ve test setlerine bölin
X_train, X_test, y_train, y_test = train_test_split(df['Lemmatized_Tweets'], df['Label'], test_size=0.2, random_state=42)

# Metin verilerini sayısal vektörlere dönüştürmek için CountVectorizer kullanın
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# SVM modelini oluşturun ve eğitin
svm_classifier = SVC(kernel='linear')  # Doğrusal SVM kullanılıyor, kernel'ı non-doğrusal yapmak için 'rbf' de kullanabilirsiniz
svm_classifier.fit(X_train_vectorized, y_train)

# Test seti üzerinde tahmin yapın
y_pred = svm_classifier.predict(X_test_vectorized)

# Doğruluk (accuracy) ve diğer metrikleri değerlendirin
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
