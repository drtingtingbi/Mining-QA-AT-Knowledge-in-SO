# File organization

Data and results of the experiment and data analysis:

1. Training data for classifier.xlsx

comparises 1165 QA-AT psost (with URL links) and 1200 non QA-AT posts collected from Stack Overflow.


- 2. AT related terms.pdf 
comprises AT names and their related terms. AT names are used for searching QA-AT posts, 
and AT related terms are used for dictionary training.  


- 3. QA related term.zip
comprises QAs and their related terms. QAs are used for eight high-level QAs identification, 
and QA related terms are used for dictionary training. 


- 4. The output of the trained dictionary.pdf 
comprises the sensitive terms of QAs and ATs in the dictionary trained by the 1741 architectural posts.  

- 5. QA-AT posts labelling and encoding.mx20 is the results of QA-AT posts labelling and encoding that were analyzed by the MAXQDA tool. We analyzed and coded the considerations discussed in QA-AT posts and the relationships between QAs and ATs. Particularly, the relationships between ATs and QAs can be found in the group named "ATs->QAs" in the coding results. The file can be opened by MAXQDA 2020, which are available at https://www.maxqda.com/ for download. You may also use the free 14-day trial version of MAXQDA 2020, which is available at https://www.maxqda.com/trial for download.

- 6. Experiment result.tet
comparises the binary classification results of six maching learning methods.

- 7. Architectural posts.xlsx
comparises the architecturel posts for dictionary training that crawled from Stack Overflow.

# Code
Crawler for archietctural posts.py 
for mining architectural posts and other potential QA-AT posts.

experiments.py
for running the classification experiments.

# Experiment environment and the used package 
Python 3.7

boto==2.49.0
boto3==1.14.20
botocore==1.17.20
certifi==2020.6.20
chardet==3.0.4
click==7.1.2
Cython==0.29.14
docutils==0.15.2
gensim==3.8.3
idna==2.10
info-gain==1.0.1
jmespath==0.10.0
joblib==0.16.0
nltk==3.5
numpy==1.19.0
pandas==1.0.5
python-dateutil==2.8.1
pytz==2020.1
regex==2020.6.8
requests==2.24.0
s3transfer==0.3.3
scikit-learn==0.23.1
scipy==1.5.1
six==1.15.0
smart-open==2.1.0
threadpoolctl==2.1.0
tqdm==4.47.0
urllib3==1.25.9
xlrd==1.2.0


