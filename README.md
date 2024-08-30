# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: TANESSHA KANNAN
RegisterNumber: 212223040225
```
```
**import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(r"C:\Users\admin\Downloads\student_scores.csv")
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)**
## Output:
![linear regression using gradient descent](sam.png)
```
## Output:

### DataSet:
![image](https://github.com/user-attachments/assets/10d5bf22-8210-45bf-ba3d-b9904a84f756)
### Head Values:
![image](https://github.com/user-attachments/assets/8ff8fc84-6d94-4876-b4e6-bd0da733738f)
### Tail Values
![image](https://github.com/user-attachments/assets/921da7ac-531a-447b-99ad-a38a04029641)
### X and Y values
![image](https://github.com/user-attachments/assets/399f5440-89e3-4d46-b1be-cbe213df0487)
### Prediction Values of X and Y
![image](https://github.com/user-attachments/assets/0f84a6b0-dec3-4a2f-9613-4aa219bdaa08)
### MSE,MAE and RMSE
![image](https://github.com/user-attachments/assets/11bfdb8b-8a19-4e7e-9891-9f779fda9ea7)
### Training Set
![image](https://github.com/user-attachments/assets/28444dd4-a2a2-4443-894b-18afbaaa75ae)
![image](https://github.com/user-attachments/assets/5f254167-6f8f-40ad-a991-e7538a4f60aa)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
