# GRIP-TASK-1-TSF
# Role - Data Science & Business Analyst Intern
# Author - Faizan Sayyed
# Task 1 Beginner Level - Prediction Using Supervised ML 
To predict the score of student studies for (9.25)Hours/Day

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# Importing all libraries required in this notebook
```
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 

# Reading data from git link
url = "https://bit.ly/3upiAre"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

# From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score

#The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).
X = s_data.iloc[:, :-1].values
Y = s_data.iloc[:,1].values

#Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

# We have split our data into training and testing sets, and now is finally the time to train our algorithm.

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train,Y_train)
print("TRAINING COMPLETE")


#PLOTTING THE REGRESSION LINE
line = clf.coef_* X+clf.intercept_

# Plotting for the test data
plt.scatter(X, Y)
plt.plot(X, line);
plt.show()

# Making Predictions
print(X_test) # TESTING DATA IN HOURS
Y_pred = clf.predict(X_test)


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
df


# NOW TESTING FOR 9.25 HOURS
hours = [[9.25]]
own_pred = clf.predict(hours)
print("NO OF HOURS = {}".format(hours))
print("PREDICTED SCORE = {}".format(own_pred[0]))


#EVALUATING THE MODEL
from sklearn import metrics
print("MEAN ABSOLUTE ERROR:", metrics.mean_absolute_error(Y_test, Y_pred))


print("TASK COMPLETED")
```



