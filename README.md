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
![image](https://github.com/DEVADARSHAN2/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119432150/f8764d0f-b615-41e5-b009-9f536c44950b)

### Head
![image](https://github.com/DEVADARSHAN2/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119432150/c1cff07a-3a3b-4ddb-91db-6555fb29d5a7)

### Tail
![image](https://github.com/DEVADARSHAN2/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119432150/d0a05ee3-5a24-476c-b5a0-e1027cf8243b)

### X and Y values
![image](https://github.com/DEVADARSHAN2/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119432150/8d311101-fe65-44b6-8b01-f3fd74ace3c9)

### Predicted values
![image](https://github.com/DEVADARSHAN2/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119432150/ed83eccd-9b14-409c-8c53-358941b039ac)

### Training set
![image](https://github.com/DEVADARSHAN2/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119432150/130f8975-c37f-4108-9d2f-d725df3ae8f7)

###  Testing set
![image](https://github.com/DEVADARSHAN2/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119432150/1f0ad825-699a-4440-96ae-04c4d6145748)

### Error calculation
![image](https://github.com/DEVADARSHAN2/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119432150/6aaf04ed-6739-4d67-a4db-b6b2b6cea827)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
