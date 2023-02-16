
'''
Bayes theorem:
tells relationship between conditional probabilities
tells that to find prob of an idea(or hypothesis or statement) given a certain evidence =
calculate prob of evidence given that idea * prior prob of idea / prior prob of evidence
P(I|E)= P(E|I)*P(I) / P(E)
---------------------------------------------------------------
NAIVE BAYES
this is classifier
for linear and non-linear relationships
returns prob of an instance belonging to a particular class

it is naive because it assumes that : for a given class, all features are independent of each other..even though they may not be
why this assumption is made.. to make it easy to compute..computationally efficient..
so it does not capture dependencies between variables...SO may not be accurate..
so it is useful to classify data having lot of dimensions..
often used in text classification problems

limitation:
it is only estimate based on assumptions.. and the actual result can vary based on uncertainity..and other factors

????
by making assumption that features are independent of each other..
then in the formulas where is this used.. are we saying p(a|b)= p(a) etc..anywhere?

if we have limited data and noisy data.. we can use this
it helps to use our prior belief about an event.. then use the info available now.. and update the belief about that event..

------------------------------------------------------
1. WHEN A,B ARE DEPENDENT,  P(A AND B) = P(B) * P(A|B)
So P(A/B)= P(A and B) / P(B)
2. BUT we can write P(A and B ) = P(A)* P(B|A)
So P(A|B) = P(A AND B)/ P(B) = P(A)*P(B|A) / P(B)
3. But We can write P(B) as P(A)*P(B/A) + P(NOT A) * P(B| NOT A)
SO it becomes
"P(A|B)
= P(A AND B) / P(B)
= P(A)* P(B/A) / P(B)
= P(A) * P(B|A) / [P(A)*P(B|A) + P(NOT A)* P(B| NOT A )]"
THIS IS BAYES THEOREM APPLICATION..
------------------------------------------------------------
1. Suppose we have a PRACTICE INTERVIEW TEST ,that is used to predict passing ACTUAL INTERVIEW.. and the test is 95% accurate.
implies that
P(passing practice interview | passed actual interview) = 95%
P(passing Practice interview | not passed actual interview)= 5%

2. If PRIOR PROB OF PASSING ACTUAL INTERVIEW is 0.05 (5%), this is our prior belief about passing actual interview..
implies that
P(passing actual interview)=5%
p( not passing actual interview)= 95%

3. what is the probability of passing Actual_Interview given a Practice_Interview test result?

We can use Bayes' theorem to calculate this:

P(passing Actual_Interview| passed Practice_Interview) = P(passing Practice_Interview| passed Actual_Interview) * P(passed Actual_Interview) / P(passed Practice_Interview)

Where:
P(passing Practice_Interview|passed Actual_Interview) = 0.95 (accuracy of the test)
P(passing Actual_Interview) = 0.05 (prior probability of having the Actual_Interview)
P(passing Practice_Interview) = P(passing Practice_Interview| passed Actual_Interview) * P(passing Actual_Interview)
+ P(passing Practice_Interview|not passing Actual_Interview) * P(not passing Actual_Interview)
= 0.95 * 0.05 + 0.05 * 0.95 = 0.0975

Substituting the values into the equation:

P(passing Actual_Interview| passed Practice_Interview) = 0.95 * 0.05 / 0.0975 = 0.4736
so even if you passed a practice interview , and that has 95% accuracy in predicting actual interview performance, the prob of passing actual interview is 47%. given that our prior belief of passing actual interview is 5%.

so we updated our prior belief..of passing actual interview.. with the evidence that we passed practice interview...
and it came to 47%..
-------------------------------------------------------------
'''


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))


# TO be practiced
#https://www.datacamp.com/tutorial/naive-bayes-scikit-learn