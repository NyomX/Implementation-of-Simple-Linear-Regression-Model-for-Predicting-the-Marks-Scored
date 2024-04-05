# EX-02 Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.

2. Set variables for assigning dataset values.

3. Import linear regression from sklearn.

4. Assign the points for representing in the graph

5. Predict the regression for marks by using the representation of the graph.

6. Compare the graphs and hence we obtained the linear regression for the given data

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sandeep V
RegisterNumber: 212223040179
```
## IMPORT REQUIRED LIBRARIES
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv("student_scores.csv")
print(df.head())
print(df.iloc[3])
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
```
## SPLITTING DATASET INTO TRAINING AND TESTING DATA
```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
```
## GRAPH PLOT FOR TRAINING DATA
```
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## GRAPH PLOT FOR TESTING DATA
```
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,y_pred,color='black')
plt.title('Hours vs Scores(Testing set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```
## CALCULATE MEAN SQUARED ERROR
```
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
#CALCULATE MEAN ABSOLUTE ERROR
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
#CALCULATE ROOT MEAN SQUARED ERROR
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
### Dataset
![image](https://github.com/NyomX/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953580/e9135290-55a7-4cbe-98fd-2748433f7a23)

### Head
![image](https://github.com/NyomX/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953580/eaf45881-4d02-4fe6-9946-edc46e39f842)

### Tail
![image](https://github.com/NyomX/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953580/982c5653-83e1-47af-924b-8dfe9d36297e)

### X and Y values
![image](https://github.com/NyomX/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953580/330317ec-6feb-4716-b3ba-2a2c8790b4ca)

### Predicted values
![image](https://github.com/NyomX/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953580/5e43890a-e2a8-4dd3-93b4-8549c3d28fca)

### Training set
![image](https://github.com/NyomX/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953580/53e0291f-b14a-4c4f-9a02-c1052a694bb8)

###  Testing set
![image](https://github.com/NyomX/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953580/d1304e63-e9af-434e-a32e-dd91f894d217)

### Error calculation
![image](https://github.com/NyomX/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145953580/57417ac2-61c3-48f3-8d01-4d711b3eb69a)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
