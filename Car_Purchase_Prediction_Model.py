'''
Predicts the car purchase amount using a multiple linear regression model based on the following independent variables:
- Gender: Categorical variable indicating the gender of the customer.
- Age: Numerical variable representing the customer's age.
- Net Worth: Numerical variable representing the customer's net worth.
- Annual Salary: Numerical variable indicating the customer's annual salary.
- Credit Card Debt: Numerical variable representing the customer's total credit card debt.

The model aims to find the relationship between these independent variables and the car purchase amount, allowing for accurate predictions.
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error

class CAR_PURCHASE_MODEL:
    def __init__(self, df):
        try:
            self.df = df
            self.X = self.df.iloc[:, 3:-1]
            self.y = self.df.iloc[:, -1]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,random_state=27)
        except Exception as e:
            error_type, error_msg, error_lineno = sys.exc_info()
            print(f'the error line number is {error_lineno.tb_lineno} --> error type is {error_type} --> error msg is {error_msg}')
    def training_data(self):
        try:
            self.reg = LinearRegression()
            self.reg.fit(self.X_train, self.y_train)
            self.train_data_prediction = self.reg.predict(self.X_train)
            self.train_data_accuracy = r2_score(self.y_train, self.train_data_prediction)
            self.training_MSE = round(sum([((i - j) ** 2) for i, j in zip(self.y_train, self.train_data_prediction)]) / len(self.y_train),2)  # without using builtin function
            self.training_RMSE = round(np.sqrt(self.training_MSE), 2)  # without using builtin function
            self.training_MAE = round(sum([(abs(i - j)) for i, j in zip(self.y_train, self.train_data_prediction)]) / len(self.y_train),2)  # without using builtin function
            print(f'the accuracy of the training data is: {self.train_data_accuracy}\nthe train_data mean squared error: {self.training_MSE}\nthe train_data mean absolute error: {self.training_MAE}\nthe train_data root mean squared error: {self.training_RMSE}')
            print('--------------------------------------------------------------------------------------')
        except Exception as e:
            error_type, error_msg, error_lineno = sys.exc_info()
            print(f'the error line number is {error_lineno.tb_lineno} --> error type is {error_type} --> error msg is {error_msg}')
    def test_data(self):
        try:
            self.reg.fit(self.X_test, self.y_test)
            self.test_data_prediction = self.reg.predict(self.X_test)
            self.test_data_accuracy = r2_score(self.y_test, self.test_data_prediction)
            self.testing_MSE = round(sum([((i - j) ** 2) for i, j in zip(self.y_test, self.test_data_prediction)]) / len(self.y_test),2)  # without using builtin function
            self.testing_RMSE = round(np.sqrt(self.testing_MSE),2)  # without using builtin function
            self.testing_MAE = round(sum([(abs(i - j)) for i, j in zip(self.y_test, self.test_data_prediction)]),2) / len(self.y_test)  # without using builtin function
            print(f'the accuracy of the test data is: {self.train_data_accuracy}\nthe test_data mean squared error: {self.testing_MSE}\nthe test_data mean absolute error: {self.testing_MAE}\nthe test_data root mean squared error: {self.testing_RMSE}')
        except Exception as e:
            error_type, error_msg, error_lineno = sys.exc_info()
            print(f'the error line number is {error_lineno.tb_lineno} --> error type is {error_type} --> error msg is {error_msg}')


if __name__ == '__main__':
    try:
        path = "C:\\Users\\abhin\\Desktop\\Sai Python practice\\Assignments\\CarPurchasing Data.csv"
        df = pd.read_csv(path, encoding='ISO-8859-1')
        cp = CAR_PURCHASE_MODEL(df)
        cp.training_data()
        cp.test_data()
    except Exception as e:
        error_type, error_msg, error_lineno = sys.exc_info()
        print(f'the error line number is {error_lineno.tb_lineno} --> error type is {error_type} --> error msg is {error_msg}')