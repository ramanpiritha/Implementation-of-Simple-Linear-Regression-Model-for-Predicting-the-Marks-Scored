# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Store data in a structured format (e.g., CSV, DataFrame).
2. Use a Simple Linear Regression model to fit the training data.
3. Use the trained model to predict values for the test set.
4. Evaluate performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Program:
```
Name: Piritharaman R
Reg no: 212223230148
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()
```

## Output:
![image](https://github.com/user-attachments/assets/ab8ccd75-c94f-465f-880a-171ab6925177)
```
df.tail()
```
## Output:
![image](https://github.com/user-attachments/assets/fa7edde5-e662-4c39-ae3b-085e13e8cbfb)
```
x=df.iloc[:,:-1].values
x
```
## Output:
![image](https://github.com/user-attachments/assets/0c8428b8-13b6-40ba-b19b-f36ee582e3a0)
```
y=df.iloc[:,1].values
y
```
## Output:
![image](https://github.com/user-attachments/assets/b9dd488d-42c9-4bb8-8d27-ed3caa0ac437)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
```
## Output:
![image](https://github.com/user-attachments/assets/bdcdbd0e-1b56-4085-8950-5e3a01613689)
```
y_test
```
## Output:
![image](https://github.com/user-attachments/assets/509946a2-87e1-46ec-82d5-af0852a2968c)
```
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE =',mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)
```
## Output:
![image](https://github.com/user-attachments/assets/89765415-a172-4c43-8b27-2f21f6372b34)
```
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/15ecf353-b7d4-40b8-9592-759d6de15f53)
```
plt.scatter(x_test,y_test,color="orange")
plt.plot(x_test,y_pred,color="red")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/bb66e6e1-649f-4115-9aa7-26f7dae686c9)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
