Methods used: 
#### Bag of words
Bag of words (BOW) cares about the frequency of the words in text, the order of words is irrelevant.
Two common ways to represent bag of words are CountVectorizer and Term Frequency, Inverse Document Frequency (TF-IDF). 
##### CountVectorizer
The CountVectorizer gives us the frequency of occurrence of words in each review. 
##### TF-IDF 
TF-IDF computes “weights” that represent how important a word is. The TF-IDF value increases proportionally to the 
number of times a word appears in a review and is offset by the number of reviews in the whole dataset (corpus) that 
contain the word. This method is used because we want to identify unique/representative words for positive reviews and 
negative reviews. More information about fit vs transform: 
https://datascience.stackexchange.com/questions/12321/whats-the-difference-between-fit-and-fit-transform-in-scikit-learn-models