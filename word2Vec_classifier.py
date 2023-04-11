import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv("comments.csv")

# Tokenize the comments

# no preprocessing

comments = [str(comment).split() for comment in data["Comments"]]

# average length of comment: 40

# manually tuned parameters
parameters = {
    'vector_size': 300,
    'window': 50,
    'min_count': 1,
    'alpha': 0.03,
    'min_alpha': 0.0007,
    'sg': 1
}

# Train the Word2Vec model
# model = Word2Vec(sentences=comments, workers=4, **parameters)
# model.save("word2vec.model")
model=Word2Vec.load("word2vec.model")

# Define a function to calculate the average vector for a sentence
def calculate_avg_vector(sentence):
    vectors = []
    for word in sentence:
        try:
            vectors.append(model.wv[word])
        except KeyError:
            pass
    if len(vectors) == 0:
        return np.zeros(100)
    else:
        return np.mean(vectors, axis=0)

# Calculate the average vector for each comment
X = [calculate_avg_vector(comment) for comment in comments]
y = data["Score"]

# Split the data into training and testing sets
mse_list = []
for i in range(5):
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Train a regression model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Test the model on the testing set
    y_pred = regressor.predict(X_test)

    # add clipping
    # 5.0 is the best score we can get
    y_pred_clipped = np.clip(y_pred, a_min=None, a_max=5.0)

    # Calculate the mean squared error
    # mse = mean_squared_error(y_test, y_pred)

    # with clipping
    mse = mean_squared_error(y_test, y_pred_clipped)

    mse_list.append(mse)

    print(f"Train/test split {i}: MSE = {mse:.4f}")

# Calculate the mean MSE across all train/test splits
mean_mse = np.mean(mse_list)
print(f"Mean MSE across all train/test splits: {mean_mse:.4f}")


# Test the model on a new comment
test_comment = "Toddy is such a horrible person, I would never take his class ever again. "
test_vector = calculate_avg_vector(test_comment.split())
predicted_score = regressor.predict([test_vector])[0]

print("Predicted score:", predicted_score)