# File organization

Data items and results of the experiment and data analysis:

1. `Training data for classifier.xlsx` comprises 1165 QA-AT posts (with URL links) and 1200 non QA-AT posts collected from Stack Overflow.


2. `AT related terms.pdf` 
comprises AT names and their related terms. AT names are used for searching QA-AT posts, 
and AT related terms are used for dictionary training.  


3. `QA related term.zip`
comprises QAs and their related terms. QAs are used for eight high-level QAs identification, 
and QA related terms are used for dictionary training. 


4. `The output of the trained dictionary.pdf` 
comprises the sensitive terms of QAs and ATs in the dictionary trained by the 1741 architectural posts.  

5. `QA-AT posts labelling and encoding.mx20` is the results of QA-AT posts labelling and encoding that were analyzed by the MAXQDA tool. We analyzed and coded the considerations discussed in QA-AT posts and the relationships between QAs and ATs. Particularly, the relationships between ATs and QAs can be found in the group named "ATs->QAs" in the coding results. The file can be opened by MAXQDA 2020, which are available at https://www.maxqda.com/ for download. You may also use the free 14-day trial version of MAXQDA 2020, which is available at https://www.maxqda.com/trial for download.

6. `Experiment result.txt`
comprises the binary classification results of six machine learning methods.

7. `Architectural posts.xlsx`
comprises the architectural posts for dictionary training that crawled from Stack Overflow.

8. `Using dic training data for classifier.xlsx`
comprises original training dataset (data item 1) and unseen but QA and AT relevant terms of the trained dictionary (date item 4). 

# Experiment replication steps

Step 1: Data pre-processing. 

* Input: Training database (Data item 1).
* Output: The pre-processed training dataset.
* Methods: (1) Tokenization, (2) Stop words removal, and (3) Stemming.
* Souce code: `experiments.py` listed in the Code section

Step 2: Training classifiers.

* Input: The pre-processed training dataset (output of Step 1).
* Feature selection methods: Word2vec and TF-IDF
* Machine learning methods: Support Vector Machine (SVM), Bayes, Decision Tree (DT), Logistic Regression (LR), Random Forest (RF), and Bagging
* Output: the binary classification of QA-AT posts and non QA-AT posts (Data item 6).
* Source code: `experiments.py` listed in the Code section

Step 3: Dictionary-based QA-AT classifier training.

* Input: Architectural posts + Training dataset (Data items 1 and 7)
* Output: Data items 4 and 6.
* Souce code: `Training the dictionary source code.py` in the Code section

 - Step 3.1. Set 1: All terms from architectural posts plus all terms from the training dataset.
    * Input: Set 1 = Architectural posts + Training dataset (Data items 1 and 7)

 - Step 3.2. Using Word2vec to train Set 1.
    * Output: vector of each term in Set 1.

 - Step 3.3. Calculating the similarities between the QA and AT terms that we manually identified (Data items 2 and 3) and the terms in Set 1.
    * Method: `word2vec.similarty by word(t)`
    * Output: similarities values between terms

 - Step 3.4. Using information gain ratio to filter terms with value > 0.350
    * Method: Information gain ratio
    * Output: the dictionary (unseen but relevant QA -AT terms) (Data item 4).

 - Step 3.5. Putting the unseen but relevant QA-AT terms to strengthen the training classifiers.
    * Step execution and the feature selection and machine learning methods are the same to Step 2.
    * Input: Data item 8.
    * Output: results of the experiment (Data item 6).



# Code
* `Crawler for architectural posts.py` for mining architectural posts and other potential QA-AT posts.

* `experiments.py`
for running the classification experiments.

* `Training the dictionary source code.py`
for training the dictionary to include more unseen but relevant QA and AT terms.

# Experiment environment and the used packages
* Python 3.7 to run our source code.

* Required packages and their versions:

`boto==2.49.0`

`boto3==1.14.20`

`botocore==1.17.20`

`certifi==2020.6.20`

`chardet==3.0.4`

`click==7.1.2`

`Cython==0.29.14`

`docutils==0.15.2`

`gensim==3.8.3`

`idna==2.10`

`info-gain==1.0.1`

`jmespath==0.10.0`

`joblib==0.16.0`

`nltk==3.5`

`numpy==1.19.0`

`pandas==1.0.5`

`python-dateutil==2.8.1`

`pytz==2020.1`

`regex==2020.6.8`

`requests==2.24.0`

`s3transfer==0.3.3`

`scikit-learn==0.23.1`

`scipy==1.5.1`

`six==1.15.0`

`smart-open==2.1.0`

`threadpoolctl==2.1.0`

`tqdm==4.47.0`

`urllib3==1.25.9`

`xlrd==1.2.0`




