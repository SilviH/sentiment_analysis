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
negative reviews. ALL THE MODELING IS BASED ON THE FREQUENCY/UNIQUENESS OF WORDS! More information about fit vs transform: 
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
##### Naive Bayes
It assumes that the presence of a particular feature is unrelated to the 
presence of any other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 3 
inches in diameter. All of these properties independently contribute to the probability that this fruit is an apple. 
So it does not consider the vectors of a review (words) together but how they contribute to certain sentiment separately.
##### Logistic Regression
The model builds a regression model to predict the probability that a given data entry belongs to the category 
numbered as “1”. Just like Linear regression assumes that the data follows a linear function, 
Logistic regression models the data using the sigmoid function.
Logistic regression becomes a classification technique only when a decision threshold is brought into the picture. 

#### Model Evaluation
##### Mean Accuracy
Used when the True Positives and True negatives are more important.
##### F1 Score
F1 Score is the weighted average of Precision and Recall. 

`F1 Score = 2*(Recall * Precision) / (Recall + Precision)`

F1 score reaches its best value at 1 and worst score at 0.Recall literally is how many of the true positives were 
recalled (found), i.e. how many of the correct hits were also found. Precision (your formula is incorrect) is how many 
of the returned hits were true positive i.e. how many of the found were correct hits.
Used when the False Negatives and False Positives are crucial.
F1 takes into account how the data is distributed, so it’s useful when you have data with imbalance classes (e.g. less
positive than negative reviews in train data or the opposite).

##### Confusion Matrix 
Table that reports the number of false positives, negatives and true positives, negatives.

|   |   |
---|---
TP | FP |
FN | TN |

