# GRIP-TASK-1-TSF
# Role - Data Science & Business Analyst Intern
# Author - Faizan Sayyed
# Task 1 Beginner Level - Prediction Using Supervised ML 
To predict the score of student studies for (9.25)Hours/Day

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Importing all libraries:
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt

# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)

