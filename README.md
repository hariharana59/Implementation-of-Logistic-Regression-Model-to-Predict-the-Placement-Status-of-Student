# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
```

Program to implement the the Logistic Regression
Model to Predict the Placement Status of Student.
Developed by: Hariharan A
RegisterNumber: 212223110013

```
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Load the California Housing dataset and select the first 3 features as input (X) and target variables (Y) (including the target price and another feature).
3. Split the data into training and testing sets, then scale (standardize) both the input features and target variables.
4. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.
5. Make predictions on the test data, inverse transform the predictions, calculate the Mean Squared Error, and print the results.
6. Stop

## Program:
```
import numpy as np 
from sklearn.datasets import fetch_california_housing 
from sklearn.linear_model import SGDRegressor 
from sklearn.multioutput import MultiOutputRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import StandardScaler

#Load the California Housing dataset 
data = fetch_california_housing()

# Use the first 3 features as inputs 
X = data.data[:, :3] # Features: 'MedInc', 'HouseAge', 'AveRooms'

# Use MedHouseVal' and 'AveOccup' as output variables 
Y = np.column_stack((data.target, data.data[:, 6])) # Targets: 'MedHouseVal', 'AveOccup'

#Split the data into training and testing sets. 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the features and target variables
scaler_X = StandardScaler() 
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test) 
Y_train = scaler_Y.fit_transform(Y_train) 
Y_test = scaler_Y.transform(Y_test)

#Initialize the SGDRegressor 
sgd = SGDRegressor(max_iter=1000, tol=1e-3)

# Use MultiOutput Regressor to handle multiple output variables 
multi_output_sgd = MultiOutputRegressor(sgd)

#Train the model 
multi_output_sgd.fit(X_train, Y_train)

# Predict on the test data 
Y_pred = multi_output_sgd.predict(X_test)

#Inverse transform the predictions to get them back to the original scale. 
Y_pred = scaler_Y.inverse_transform(Y_pred) 
Y_test = scaler_Y.inverse_transform(Y_test)

#Evaluate the model using Mean Squared Error 
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

#Optionally, print some predictions 
print("\nPredictions: \n", Y_pred[:5]) # Print first 5 predictions
```

## Output:



![image](https://github.com/user-attachments/assets/d5440996-7f77-487e-a12f-dece8a9fec9c)

![image](https://github.com/user-attachments/assets/d97bc6e1-4fe4-4234-80d0-aaa9aa897378)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
