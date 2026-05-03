from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Basit veri (spam / normal mesajlar)
texts = [
    "Win money now",
    "Limited offer just for you",
    "Call me later",
    "Let's meet tomorrow",
    "Free cash prize",
    "Hello how are you"
]

labels = [1, 1, 0, 0, 1, 0]  # 1 = spam, 0 = normal

# Metni sayılara çevir
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Model oluştur ve eğit
model = MultinomialNB()
model.fit(X, labels)

# Test
test_message = ["Free money now"]
X_test = vectorizer.transform(test_message)

prediction = model.predict(X_test)

print("Message:", test_message[0])
print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")
