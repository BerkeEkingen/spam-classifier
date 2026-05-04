from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)


texts =texts = [
    "Win money now",
    "Limited offer just for you",
    "Free cash prize",
    "Call me later",
    "Let's meet tomorrow",
    "Hello how are you",
    "Congratulations you won lottery",
    "Click this link to claim prize",
    "Free iPhone waiting for you",
    "Your account has won a reward",
    "Are you coming tonight?",
    "See you later bro",
    "Can we meet at school?",
    "Don't forget the homework"
]

labels = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)


HTML = """
<!doctype html>
<title>Spam Classifier</title>
<h2>Spam Detector</h2>
<form method=post>
  <input type=text name=message placeholder="Write message">
  <input type=submit value=Check>
</form>
{% if prediction %}
<p><b>Result:</b> {{ prediction }}</p>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        msg = request.form["message"]
        msg_vec = vectorizer.transform([msg])
        result = model.predict(msg_vec)[0]
        prediction = "Spam" if result == 1 else "Not Spam"
    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
   import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
