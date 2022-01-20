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

#### Supervised learning
Models are trained using labeled data (data that has already sentiment information).
Two common types of supervised learning are **Regression** used to predict continuous values such as price, salary, age 
and **Classification** used to predict discrete values such as male/female, spam/not spam, positive/negative.

#### Classification methods
##### Support Vector Machine
The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space that distinctly 
separates the two classes of data points.
##### Decision Trees
Creates yes/no questions and continually splits the dataset. Every time you answer a question, you’re creating branches 
and segmenting the dataset into regions - nodes. Last nodes created are called leaf nodes. The goal is to continue to 
splitting the dataset, until you don’t have any more rules to apply or no data points (reviews) left. Then, it’s time 
to assign a class to all data points in each leaf node. Most times, you end up with mixed leaf nodes, where not all 
data points have to the same class (contains both positive and negative reviews).
In the end, the algorithm can only assign one class to the data points in each leaf node. Future data point will be 
classified based on in which leaf node it falls into.
