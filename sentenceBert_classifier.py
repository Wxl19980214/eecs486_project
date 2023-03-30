import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# Load the data
df = pd.read_csv('comments.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Comments'], df['Score'], test_size=0.2, random_state=42)

# Apply Sentence-BERT to convert the comments into numerical embeddings
model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
X_train_emb = model.encode(X_train)
X_test_emb = model.encode(X_test)

# Train a logistic regression classifier on the embeddings
clf = LogisticRegression(random_state=42)
clf.fit(X_train_emb, y_train)

# Test the classifier on a new comment
new_comment = 'This is a great product!'
new_comment_emb = model.encode([new_comment])
predicted_score = clf.predict(new_comment_emb)[0]

print(predicted_score)
