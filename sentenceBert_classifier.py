import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import torch
print(torch.cuda.is_available())
# Load the data
df = pd.read_csv('comments.csv')
# iterate over the rows and change a variable
for index, row in df.iterrows():
    df.at[index,'Comments']=str(row["Comments"])
model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2',device="cuda")

X=df["Comments"].values
y=df["Score"].values

X_encoded=model.encode(X)
print("checkpoint1")
# Split the data into training and testing sets using 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_list=[]

#LinearRegression
# for fold, (train_index, test_index) in enumerate(kf.split(X_encoded)):
#     print(f'Fold {fold + 1}')
#     X_train, X_test = X_encoded[train_index], X_encoded[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     # Perform linear regression and evaluate the model using the training and testing sets
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)
#     y_test_pred=np.clip(y_test_pred, a_min=None, a_max=5.0)
#     print(f'Test MSE at iterartion {fold+1} : {mean_squared_error(y_test, y_test_pred)}')
#
#
#     mse_list.append(mean_squared_error(y_test, y_test_pred))

#SVM
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


mean_mse = np.mean(mse_list)
print(f"Mean MSE across all train/test splits: {mean_mse:.4f}")