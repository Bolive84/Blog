Title: Introduction To Word Embeddings - How To Train Your Own Word Vectors Using The Simpsons Dialogues?
Date: 2019-10-07
Slug: Blog_3
<<<<<<< HEAD
cover: https://cdn.pixabay.com/photo/2018/07/03/07/09/block-chain-3513216_960_720.jpg
=======
cover: /assets/images/article_cover.jpg
>>>>>>> f72aa99d6090b9169cad31621d623daad9052776

<font color='darkblue'>  
<h2> <center>Introduction To Word Embeddings</center>  <br>  <center> <i>How To Train Your Own Word Vectors Using The Simpsons Dialogues? </i> </center>
</h2>
</font>


<font color='darkblue'> <h3> Why Word Vectors? </h3> </font>

As the volume of textual data generated on the web or uploaded in large servers around the world is growing in an exponential way, the need to analyze this type of data is becoming more and more crucial to understand trends, people behaviors, purchase intentions, and more generally the world we live in.

In order to increase our ability to analyse relationships across words, sentences and documents, words can be transformed into vectors. In physics, vectors are arrows, in computer science and statistics, vectors are columns of values, like one numeric series in a dataframe. A word vector is literally a way for us to represent words as vectors, allowing us to use machine learning to better comprehend the structure and content of language by organizing concepts and learning the relationships among them.

In this tutorial, I am going to describe the different steps to follow to train a `word2vec` model with your own text-based dataset. Word2vec was created and published by Tomas Mikolov and his research team at Google in 2013. This tool provides an efficient implementation of the continuous bag-of-words (CBOW) and skip-gram architectures for computing vector representations of words -also called word embeddings-, using a two-layer neural network. It takes in observations (sentences, tweets, books), first constructs a vocabulary from the training text data, and then learns vector representation of words. The resulting word vector file can be used as features in many natural language processing and machine learning applications.


Here is a quick summary of the tutorial:

1. Import data and libraries
2. Pre-process text
3. Transform text into vectors with `word2vec` and train the model
4. Explore the model
5. Conclusion

<font color='darkblue'> <h4> My text dataset:</h4></font>

I could find on Kaggle this dataset containing script lines for approximately 600 Simpsons episodes, dating back to 1989.
The `.csv` file is about 9MB, and includes more than 150K lines of dialogues between the different characters of the series, data is organized in 2 columns, each row showing the character's name and the text actually spoken, A-MA-ZING!

File can be found here: https://www.kaggle.com/pierremegret/dialogue-lines-of-the-simpsons

![png](images/Blog_3/Simpsons_magnifying.png)

<font color='darkblue'> <h3> 1. Import data and libraries </h3> </font>


```python
# Install/upgrade Gensim if needed:
# !pip install gensim --upgrade

# Import Gensim and data science standard libraries:
import gensim # to get access to word2vec tool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Import Simpsons dataset, and read it in a pandas dataframe:
df = pd.read_csv('simpsons_dataset.csv')
```


```python
# let's explore quickly the data:

# Data is showing as text strings, contains 2 columns and 158314 rows, and has some null values in each of the 2 columns:
print('Dataframe information: ')
print()
df.info()
```

    Dataframe information:

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 158314 entries, 0 to 158313
    Data columns (total 2 columns):
    raw_character_text    140500 non-null object
    spoken_words          131855 non-null object
    dtypes: object(2)
    memory usage: 2.4+ MB



```python
# Data looks great!
print('Dataframe top rows:')
df.head()
```

    Dataframe top rows:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>raw_character_text</th>
      <th>spoken_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Miss Hoover</td>
      <td>No, actually, it was a little of both. Sometim...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lisa Simpson</td>
      <td>Where's Mr. Bergstrom?</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Miss Hoover</td>
      <td>I don't know. Although I'd sure like to talk t...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lisa Simpson</td>
      <td>That life is worth living.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Edna Krabappel-Flanders</td>
      <td>The polls will be open from now until the end ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Here is the total of null values per columns:
print('Number of null values in dataframe: ')
df.isnull().sum()
```

    Number of null values in dataframe:





    raw_character_text    17814
    spoken_words          26459
    dtype: int64




```python
# Dataset is large enough to run our demo, so let's get rid of all null values:
df = df.dropna()
```


```python
# Just making sure that our dataset is free of nulls now:
print('Number of null values remaining: ')
df.isnull().sum()
```

    Number of null values remaining:





    raw_character_text    0
    spoken_words          0
    dtype: int64




```python
# After dropping the nulls, 131853 rows remain in the dataset:
print('Dataframe shape:', df.shape)
```

    Dataframe shape: (131853, 2)



```python
# Let's rename the 2 columns:
df.rename(columns={'raw_character_text': 'character', 'spoken_words': 'sequence'}, inplace=True)
```

<font color='darkblue'> <h3> 2. Pre-process text </h3> </font>

Pre-preprocessing is needed for transferring text from human language to machine-readable format for further processing.


```python
# Import text nltk pre-processing tools:
from nltk.stem import WordNetLemmatizer # to lemmatize words
from nltk.tokenize import RegexpTokenizer # to'tokenize' words
from nltk.corpus import stopwords # to remove stopwords.
import re # to access to regular expression matching operations

def preprocess (x): # create a pre-processing function:
    string_text = str(x) # convert text to strings
    lower_case = string_text.lower() # lowercase text
    lower_case = re.sub(r'[^a-zA-Z]', ' ', lower_case) # remove non-alphabetic characters
    retokenizer = RegexpTokenizer(r'\w+')
    words = retokenizer.tokenize(lower_case) # split strings into substrings using a regular expression
    lemmer = WordNetLemmatizer() # take words and attempt to return their base/dictionary form.
    stops = set(stopwords.words('english')) # remove english stopwords, some example of stopwords are as 'what', 'as', 'of', 'into'
    meaningful_words = [w for w in words if not w in stops]
    if len(meaningful_words) >2: # Let's remove sentences that are less than two words long as Word2Vec uses context words to learn the vector representation of a target word
        return " ".join([lemmer.lemmatize(word) for word in meaningful_words if not word in stops])
```


```python
# Apply our function to the Simpsons dialogues:
df.sequence = df.sequence.apply(preprocess)
```


```python
# Pre-processing is now done, but sentences that were 1 or 2 words long have been removed from our dataset, see 'sequence' at row [1]:
print('Pre-processed dataframe top rows:')
df.head()
```

    Pre-processed dataframe top rows:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>character</th>
      <th>sequence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Miss Hoover</td>
      <td>actually little sometimes disease magazine new...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lisa Simpson</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Miss Hoover</td>
      <td>know although sure like talk touch lesson plan...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lisa Simpson</td>
      <td>life worth living</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Edna Krabappel-Flanders</td>
      <td>poll open end recess case decided put thought ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 38061 new nulls appear throughout our preprocessed dataset:
print('Number of null values in pre-processed dataframe: ', df.sequence.isnull().sum())
```

    Number of null values in pre-processed dataframe:  38061



```python
# ..so let's remove them as well as all duplicates:
df_clean = df.dropna().drop_duplicates()
```


```python
# Our 'clean' dataset includes now 93317 rows of pre-processed text:
print('Shape of the clean version of our dataframe: ', df_clean.shape)
```

    Shape of the clean version of our dataframe:  (93317, 2)



```python
# Let's use Gensim Phrases package to automatically detect bigrams (common phrases) from a list of sentences.
# This will connect "bart" and "simpson" when they appear side by side, so the model will treat 'homer_simpson' as one word
from gensim.models.phrases import Phrases, Phraser # Phrases() takes a list of list of words as input:

sent = [row.split() for row in df_clean['sequence']]
phrases = Phrases(sent, min_count=20) # we will ignore all bigrams with total collected count lower than 20.
bigram = Phraser(phrases)
sentences = bigram[sent]

print('First 15 bigrams in alpha order:' )
pd.DataFrame(sorted(bigram.phrasegrams)[:15], columns=['first_word', 'second_word'])

# We can see at row 12 that when 'bart' and 'simpson' show up side by side in a sentence,
# they will be now considered as one word, the bigram 'bart_simpson'
```

    First 15 bigrams in alpha order:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>first_word</th>
      <th>second_word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b'across'</td>
      <td>b'street'</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b'always'</td>
      <td>b'wanted'</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b'amusement'</td>
      <td>b'park'</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b'angry'</td>
      <td>b'dad'</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b'answer'</td>
      <td>b'question'</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b'anyone'</td>
      <td>b'else'</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b'ask'</td>
      <td>b'question'</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b'aunt'</td>
      <td>b'selma'</td>
    </tr>
    <tr>
      <th>8</th>
      <td>b'aw'</td>
      <td>b'geez'</td>
    </tr>
    <tr>
      <th>9</th>
      <td>b'b'</td>
      <td>b'c'</td>
    </tr>
    <tr>
      <th>10</th>
      <td>b'ba'</td>
      <td>b'ba'</td>
    </tr>
    <tr>
      <th>11</th>
      <td>b'bad'</td>
      <td>b'news'</td>
    </tr>
    <tr>
      <th>12</th>
      <td>b'bart'</td>
      <td>b'simpson'</td>
    </tr>
    <tr>
      <th>13</th>
      <td>b'best'</td>
      <td>b'friend'</td>
    </tr>
    <tr>
      <th>14</th>
      <td>b'big'</td>
      <td>b'deal'</td>
    </tr>
  </tbody>
</table>
</div>




```python
from collections import defaultdict # defaultdict will allow us to count the number of occurrences for each word.

word_count = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_count[i] += 1

print("Number of words in our dataframe: ", len(word_count))
```

    Number of words in our dataframe:  32943



```python
# What are the most frequent words appearing in our data?
print('Most frequent words:')
pd.DataFrame(sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:15], columns=['word', 'frequency'])

# Homer and Bart are respectively the 10th and 12th most frequently used word in the show,
# which makes sense as they are the 2 main characters of the series
```

    Most frequent words:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>oh</td>
      <td>6726</td>
    </tr>
    <tr>
      <th>1</th>
      <td>well</td>
      <td>5425</td>
    </tr>
    <tr>
      <th>2</th>
      <td>like</td>
      <td>4941</td>
    </tr>
    <tr>
      <th>3</th>
      <td>get</td>
      <td>4924</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>4756</td>
    </tr>
    <tr>
      <th>5</th>
      <td>know</td>
      <td>4593</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hey</td>
      <td>3754</td>
    </tr>
    <tr>
      <th>7</th>
      <td>right</td>
      <td>3535</td>
    </tr>
    <tr>
      <th>8</th>
      <td>got</td>
      <td>3420</td>
    </tr>
    <tr>
      <th>9</th>
      <td>homer</td>
      <td>3040</td>
    </tr>
    <tr>
      <th>10</th>
      <td>go</td>
      <td>2999</td>
    </tr>
    <tr>
      <th>11</th>
      <td>want</td>
      <td>2888</td>
    </tr>
    <tr>
      <th>12</th>
      <td>bart</td>
      <td>2802</td>
    </tr>
    <tr>
      <th>13</th>
      <td>think</td>
      <td>2687</td>
    </tr>
    <tr>
      <th>14</th>
      <td>time</td>
      <td>2577</td>
    </tr>
  </tbody>
</table>
</div>



<font color='darkblue'> <h3> 3. Transform text into vectors with `word2vec` and train the model </h3> </font>

After training, `word2vec` can be used to map each word to a vector of typically several hundred elements, which represent that word’s relation to other words. This vector is the neural network’s hidden layer.


```python
from gensim.models import Word2Vec # import word2vec from Gensim library

w2v_model = Word2Vec(
                     size=300, # number of dimensions of our word vectors
                     window=10, # set the number of "context words" at 10
                     min_count=20, # ignore all words with total frequency lower than 20
                     sample=0.0001, # set the threshold for configuring which higher-frequency words are randomly downsampled
                     negative=20, # set the number of noise words to be drawn at 20
                     workers=4) # number of threads used in parallel to train the machine
```


```python
# This step will build the vocabulary table:
w2v_model.build_vocab(sentences)
```


```python
# Model can now be trained:
w2v_model.train(
    sentences,  
    total_examples=w2v_model.corpus_count, # number of sentences in the corpus
    epochs=30 # number of iterations in the corpus
)
```




    (8742042, 18370410)




```python
# init_sims will precompute L2-normalized vectors and ,if true, will forget the original vectors and only keep the normalized ones, this can save lots of memory
w2v_model.init_sims(replace=False)
```

<font color='darkblue'> <h3> 4. Explore the model </h3> </font>

Let's see what our model finds when we ask it to pull the top-10 most similar words to the main characters of the show with `most_similar` method:


```python
print('Top 10 most similar words to Homer:')
pd.DataFrame(w2v_model.wv.most_similar(positive=["homer"]), columns=['word', 'frequency'])
```

    Top 10 most similar words to Homer:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>marge</td>
      <td>0.647157</td>
    </tr>
    <tr>
      <th>1</th>
      <td>becky</td>
      <td>0.551345</td>
    </tr>
    <tr>
      <th>2</th>
      <td>lenny_carl</td>
      <td>0.499023</td>
    </tr>
    <tr>
      <th>3</th>
      <td>creepy</td>
      <td>0.498600</td>
    </tr>
    <tr>
      <th>4</th>
      <td>homie</td>
      <td>0.497142</td>
    </tr>
    <tr>
      <th>5</th>
      <td>husband</td>
      <td>0.485995</td>
    </tr>
    <tr>
      <th>6</th>
      <td>soul_mate</td>
      <td>0.483098</td>
    </tr>
    <tr>
      <th>7</th>
      <td>crummy</td>
      <td>0.481756</td>
    </tr>
    <tr>
      <th>8</th>
      <td>leave_alone</td>
      <td>0.469040</td>
    </tr>
    <tr>
      <th>9</th>
      <td>sometime</td>
      <td>0.468563</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Top 10 most similar words to Bart:')
pd.DataFrame(w2v_model.wv.most_similar(positive=["bart"]), columns=['word', 'frequency'])
```

    Top 10 most similar words to Bart:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>lisa</td>
      <td>0.745548</td>
    </tr>
    <tr>
      <th>1</th>
      <td>maggie</td>
      <td>0.613382</td>
    </tr>
    <tr>
      <th>2</th>
      <td>learned_lesson</td>
      <td>0.602921</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mom_dad</td>
      <td>0.602509</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pay_attention</td>
      <td>0.596564</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mom</td>
      <td>0.589715</td>
    </tr>
    <tr>
      <th>6</th>
      <td>homework</td>
      <td>0.586715</td>
    </tr>
    <tr>
      <th>7</th>
      <td>tell_truth</td>
      <td>0.582463</td>
    </tr>
    <tr>
      <th>8</th>
      <td>feel_better</td>
      <td>0.581206</td>
    </tr>
    <tr>
      <th>9</th>
      <td>dr_hibbert</td>
      <td>0.581166</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Top 10 most similar words to Lisa:')
pd.DataFrame(w2v_model.wv.most_similar(positive=["lisa"]), columns=['word', 'frequency'])
```

    Top 10 most similar words to Lisa:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bart</td>
      <td>0.745548</td>
    </tr>
    <tr>
      <th>1</th>
      <td>learned_lesson</td>
      <td>0.603608</td>
    </tr>
    <tr>
      <th>2</th>
      <td>homework</td>
      <td>0.603168</td>
    </tr>
    <tr>
      <th>3</th>
      <td>maggie</td>
      <td>0.574106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mom</td>
      <td>0.568127</td>
    </tr>
    <tr>
      <th>5</th>
      <td>grownup</td>
      <td>0.563802</td>
    </tr>
    <tr>
      <th>6</th>
      <td>saxophone</td>
      <td>0.561525</td>
    </tr>
    <tr>
      <th>7</th>
      <td>surprised</td>
      <td>0.559258</td>
    </tr>
    <tr>
      <th>8</th>
      <td>daughter</td>
      <td>0.558765</td>
    </tr>
    <tr>
      <th>9</th>
      <td>math</td>
      <td>0.550277</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Top 10 most similar words to Marge:')
pd.DataFrame(w2v_model.wv.most_similar(positive=["marge"]), columns=['word', 'frequency'])
```

    Top 10 most similar words to Marge:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>homer</td>
      <td>0.647157</td>
    </tr>
    <tr>
      <th>1</th>
      <td>husband</td>
      <td>0.601620</td>
    </tr>
    <tr>
      <th>2</th>
      <td>homie</td>
      <td>0.543166</td>
    </tr>
    <tr>
      <th>3</th>
      <td>becky</td>
      <td>0.541192</td>
    </tr>
    <tr>
      <th>4</th>
      <td>marriage</td>
      <td>0.536475</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ashamed</td>
      <td>0.531238</td>
    </tr>
    <tr>
      <th>6</th>
      <td>tell_truth</td>
      <td>0.512526</td>
    </tr>
    <tr>
      <th>7</th>
      <td>disappointed</td>
      <td>0.511782</td>
    </tr>
    <tr>
      <th>8</th>
      <td>fault</td>
      <td>0.506138</td>
    </tr>
    <tr>
      <th>9</th>
      <td>brunch</td>
      <td>0.499643</td>
    </tr>
  </tbody>
</table>
</div>



It looks like all this information makes sense! We can use `doesnt_match` method to find which word from the given list doesn’t go with the others:


```python
print('Word to exclude from the list:', w2v_model.wv.doesnt_match(['bart', 'lisa', 'maggie', 'milhouse']))
```

    Word to exclude from the list: milhouse


    C:\Users\Administrator\Anaconda3\lib\site-packages\gensim\models\keyedvectors.py:877: FutureWarning: arrays to stack must be passed as a "sequence" type such as list or tuple. Support for non-sequence iterables such as generators is deprecated as of NumPy 1.16 and will raise an error in the future.
      vectors = vstack(self.word_vec(word, use_norm=True) for word in used_words).astype(REAL)



```python
print('Word to exclude from the list:', w2v_model.wv.doesnt_match(['moe', 'lenny', 'carl', 'homer', 'marge']))
```

    Word to exclude from the list: marge


In physics, we can add/subtract vectors to understand how two forces might act on an object. Let's see if we can use `most_similar` method to do the same thing with word vectors.

What happens when adding 'woman' to 'homer' and substract 'man?


```python
w2v_model.wv.most_similar(positive=["woman", "homer"], negative=["man"], topn=10)
```




    [('marge', 0.5261433124542236),
     ('husband', 0.4614260196685791),
     ('homie', 0.449959933757782),
     ('brunch', 0.4463199973106384),
     ('wife', 0.44602805376052856),
     ('marriage', 0.44539350271224976),
     ('grownup', 0.44457897543907166),
     ('wasted', 0.4415558874607086),
     ('affair', 0.4279390275478363),
     ('luann', 0.4259093403816223)]



This is correct, Marge is Homer female counterpart!

<font color='darkblue'> <h3> 5. Conclusion </h3> </font>

In this quick introduction to word embeddings I tried to describe as simply as possible the different steps you will need to go through if you want to train a word vectorizer with your own data. In this example, I chose to use Google's `word2vec` method, but other vectorizers can be available on sklearn's library as `CountVectorizer` , `HashingVectorizer` or `TfidfVectorizer`. Word embedding provides to machines much more information about words than has previously been possible using traditional representations of words. Word vectors are essential for solving NLP problems such as speech recognition, sentiment analysis, named entity recognition, spam filtering, and machine translation, they are an amazingly powerful concept and applications in this field are practically infinite. The importance of word embeddings in deep learning becomes more and more evident by looking at the number of researches in the field, so I hope that this tutorial was useful for you and helped you to better understand the mechanisms of these methods.

![jpg](images/Blog_3/simpsons460.jpg)
