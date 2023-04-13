import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor



print(torch.cuda.is_available())
# Load the data
df = pd.read_csv('comments.csv',na_values=" NaN")
# iterate over the rows and change a variable
for index, row in df.iterrows():
    df.at[index,'Comments']=str(row["Comments"])
    if(row["Attendance"]==" Mandatory"):
        df.at[index,"Attendance"]=1
    elif (row["Attendance"]==" Not Mandatory"):
        df.at[index,"Attendance"]=-1
    else:
        df.at[index,"Attendance"]=0
    if(row["Would Take Again"]==" Yes"):
        df.at[index,"Would Take Again"]=1
    elif (row["Would Take Again"]==" No"):
        df.at[index,"Would Take Again"]=-1
    else:
        df.at[index,"Would Take Again"]=0
model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2',device="cuda")

X=df["Comments"].values
y=df["Score"].values
attendance=df["Attendance"].values
wouldTakeAgain=df["Would Take Again"].values

X_encoded=model.encode(X)
X_encoded = np.concatenate((X_encoded, attendance.reshape(-1, 1)), axis=1)
X_encoded= np.concatenate((X_encoded, wouldTakeAgain.reshape(-1, 1)), axis=1)
print("checkpoint1")
# Split the data into training and testing sets using 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mse_list=[]

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

#SVM
# for fold, (train_index, test_index) in enumerate(kf.split(X_encoded)):
#     print(f'Fold {fold + 1}')
#     X_train, X_test = X_encoded[train_index], X_encoded[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#
#     model = SVR(kernel='linear')
#     model.fit(X_train, y_train)
#
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)
#     y_test_pred=np.clip(y_test_pred, a_min=None, a_max=5.0)
#     print(f'Test MSE at iterartion {fold+1} : {mean_squared_error(y_test, y_test_pred)}')
#
#
#     mse_list.append(mean_squared_error(y_test, y_test_pred))


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

mean_mse = np.mean(mse_list)
print(f"Mean MSE across all train/test splits: {mean_mse:.4f}")