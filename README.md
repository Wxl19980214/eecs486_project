# eecs486_project

A survey project running different combinations of text embedding models and regression models. 

The goal is to predict a student's overall rating toward a professor based on his/her comment.

# Requirements

To run our project, a `python >= 3.6` is required for Word2Vec and a GPU is required for Sentence-BERT.

All packages needed is within requirements.txt, simply do `pip install -r requirements.txt` to install them.

# Project Overview

### Data collection

The crawler.py is used to collect data needed for this project. 

run `python3 crawler.py` to collect data and save it to the `comments.csv` file.

We recommand using the exisiting `comments.csv` file in the repo because crawler.py take times to run.

### Text Embedding and Regression

We have two files, `sentenceBert_classifier.py` and `word2Vec_classifier.py`. each file represents

the two most cutting edge text embedding models sentence-BERT and Word2Vec respectively. 

Use `python3 sentenceBert_classifier.py` to run the file.

Within each file, there are three different regression methods written: Linear Regression, 

Support Vector Machine (svm), and Random Forest Regression. We will use linear regression as

our default method and baseline performance for our experiments. We also not recommanding 

running the other two methods because due to the complexity of the models, it takes a very

long time to train and generate final result.

# Result

The result of our experiment is in the excel file called `result.excel`.

