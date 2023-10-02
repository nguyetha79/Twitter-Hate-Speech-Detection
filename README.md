# Twitter-Hate-Speech-Detection

### Installation
---

- First of all, we install the Hugging Face `datasets` library, which is popular which is a popular library for working with various natural language processing (NLP) datasets. 

- In addition, the `tweet-preprocessor` library provides a set of functions and utilities for preprocessing and cleaning text data specifically tailored for tweets and social media text. 

```python
!pip install datasets
!pip install tweet-preprocessor
```

### Loading Packages
---

- Then we load some EDA packages such as: 
    - `load_datasets`: loads and access various NLP datasets

    - `numpy`: a fundamental library used for tasks involving arrays and matrices 
    
    - `pandas`: a powerful library for data manipulation and analysis
 
```python
from datasets import load_dataset
import pandas as pd
import numpy as np
```

- Next, we import several Python libraries for data visualization and word cloud generation. 
     - `seaborn`: draws attractive and informative statistical graphics

     - `wordcloud`: generates word clouds, visual representations of text data where words are displayed in varying sizes, with more frequent words appearing larger

     - `matplotlib.pyplot`: creates plots and charts

     - `style`: sets the visual style of our plots

```python
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from wordcloud import WordCloud
```

- Above all, the task of model building cannot be finished without machine learning libraries. We will use several modules and classes from the `scikit-learn` library. 

    - `CountVectorizer`: converts a given text into a vector based on the number of times each word appears across the entire text. 
    
    - `train_test_split`: splits a dataset into training and testing subsets. 
    
    - `cross_validate`: estimates the performance of machine learning models.
 
    - `GridSearchCV`:  the process of performing hyperparameter tuning
    
   - `DecisionTreeClassifier`: an algorithm for building decision tree-based classification models. 
   
   - `accuracy_score`: computes the accuracy of classification models. 
   
   - `classification_report`: a text report that includes precision, recall, F1-score 
   
   - `confusion_matrix`: computes a confusion matrix, which is displayed in a visually informative way with the help of `ConfusionMatrixDisplay`

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
```

- Last but not least, we import some modules and set up text preprocessing tools. 
    - `Re`: used for pattern matching and text manipulation. 
    
    - `preprocessor`: useful when working with text data from social media platforms like Twitter.

    - `string`: provides a collection of string constants and functions for text processing. 
    
    - `nltk.SnowballStemmer("english")`: the process of reducing words to their root form.
    
    - Next we download `stopwords` dataset and define a set containing the English stopwords obtained from the NLTK corpus.  

```python
import re
import preprocessor as p
import string
import nltk
from nltk.util import pr
stemmer = nltk.SnowballStemmer("english")
nltk.download('stopwords')
from nltk.corpus import stopwords
stopword = set(stopwords.words("english"))
```

