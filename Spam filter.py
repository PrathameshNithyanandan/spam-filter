import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("/Users/prathameshnithyanandan/Documents/Project 1/spam.csv", encoding='latin-1')[['v1','v2']]
df.columns = ['label','message']

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Ham','Spam']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

new_message = "you have won 400 dollars, send your phone number"
new_vector = vectorizer.transform([new_message])
prediction = model.predict(new_vector)[0]
print("\nNew message prediction:", le.inverse_transform([prediction])[0])
