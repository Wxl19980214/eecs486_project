import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load the data
data = pd.read_csv("comments.csv",na_values=" NaN")
for index, row in data.iterrows():
    data.at[index,'Comments']=str(row["Comments"])
    if(row["Attendance"]==" Mandatory"):
        data.at[index,"Attendance"]=1
    elif (row["Attendance"]==" Not Mandatory"):
        data.at[index,"Attendance"]=-1
    else:
        data.at[index,"Attendance"]=0
    if(row["Would Take Again"]==" Yes"):
        data.at[index,"Would Take Again"]=1
    elif (row["Would Take Again"]==" No"):
        data.at[index,"Would Take Again"]=-1
    else:
        data.at[index,"Would Take Again"]=0
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
model = Word2Vec(sentences=comments, workers=4, **parameters)
# model.save("word2vec.model")
# model=Word2Vec.load("word2vec.model")

# Define a function to calculate the average vector for a sentence
def calculate_avg_vector(sentence):
    vectors = []
    for word in sentence:
        try:
            vectors.append(model.wv[word])
        except KeyError:
            pass
    if len(vectors) == 0:
        return np.zeros(300)
    else:
        return np.mean(vectors, axis=0)

# Calculate the average vector for each comment
attendance=data["Attendance"].values
wouldTakeAgain=data["Would Take Again"].values
X_encoded = np.array([calculate_avg_vector(comment) for comment in comments])
X_encoded = np.concatenate((X_encoded, attendance.reshape(-1, 1)), axis=1)
X_encoded= np.concatenate((X_encoded, wouldTakeAgain.reshape(-1, 1)), axis=1)

y = data["Score"]

# Split the data into training and testing sets
mse_list = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#LinearRegression
for fold, (train_index, test_index) in enumerate(kf.split(X_encoded)):
    print(f'Fold {fold + 1}')
    X_train, X_test = X_encoded[train_index], X_encoded[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Perform linear regression and evaluate the model using the training and testing sets
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_pred=np.clip(y_test_pred, a_min=None, a_max=5.0)
    print(f'Test MSE at iterartion {fold+1} : {mean_squared_error(y_test, y_test_pred)}')


    mse_list.append(mean_squared_error(y_test, y_test_pred))


# svm
'''
for fold, (train_index, test_index) in enumerate(kf.split(X_encoded)):
    print(f'Fold {fold + 1}')
    X_train, X_test = X_encoded[train_index], X_encoded[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVR(kernel='linear')
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_pred=np.clip(y_test_pred, a_min=None, a_max=5.0)
    print(f'Test MSE at iterartion {fold+1} : {mean_squared_error(y_test, y_test_pred)}')


    mse_list.append(mean_squared_error(y_test, y_test_pred))
'''


#RandomForest
# for fold, (train_index, test_index) in enumerate(kf.split(X_encoded)):
#     print(f'Fold {fold + 1}')
#     X_train, X_test = X_encoded[train_index], X_encoded[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#
#     forest_parameters={
#         'n_estimators':300,
#         'max_depth':15,
#         'min_samples_split':2,
#         'min_samples_leaf':1,
#         'random_state':42
#     }
#     # Perform linear regression and evaluate the model using the training and testing sets
#     model = RandomForestRegressor(**forest_parameters)
#     model.fit(X_train, y_train)
#
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)
#     y_test_pred=np.clip(y_test_pred, a_min=None, a_max=5.0)
#     print(f'Test MSE at iterartion {fold+1} : {mean_squared_error(y_test, y_test_pred)}')
#
#
#     mse_list.append(mean_squared_error(y_test, y_test_pred))


# Calculate the mean MSE across all train/test splits
mean_mse = np.mean(mse_list)
print(f"Mean MSE across all train/test splits: {mean_mse:.4f}")


