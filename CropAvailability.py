import math
import statistics
import csv
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



def load_data(fileName,x, y):
    '''#Extracts data from csv file and assigns it to the year associated'''

    mydict = {}
    with open(fileName) as numbers:
        numbers_data = csv.reader(numbers, delimiter=',')
        next(numbers_data) #skips the headers

        for i in range(x):
            next(numbers_data, None)


        for i, row in enumerate(numbers_data, start=x):
            if i <= y:
                mydict[row[1]] = float(row[3])

            else:
                break


    return mydict

#Function to plot preexisting data
def plot_dict(dictionary):
    '''Graphs the availability data'''


    # Extract keys and values from the dictionary
    x_values = list(dictionary.keys())
    y_values = list(dictionary.values())

    # Plot the data
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    plt.xlabel('Year')
    plt.ylabel('Food Availability')
    plt.title('Projected Food Availability')
    plt.xticks(np.arange(0, 61, 10))
    plt.grid(False)
    plt.show()


def univariate_calculation(Data):
    '''Predict the average availability of the fruits and vegetables'''

    df = pd.DataFrame(list(Data.items()), columns=['Year', 'Value'])
    # Convert 'Year' column to datetime
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    # Set 'Year' column as index
    df.set_index('Year', inplace=True)
    # Fit ARIMA model
    model = ARIMA(df['Value'], order=(5, 1, 0))
    model_fit = model.fit()

    future_dates_10_years = pd.date_range(start=df.index[-1], periods=10, freq='YE') + pd.DateOffset(years=1)
    forecast_10_years = model_fit.forecast(steps=len(future_dates_10_years))
    list_of_forcasted_values = forecast_10_years.values

    return forecast_10_years, list_of_forcasted_values

def error_calculation(Data):

    '''Finds the error calculation percentage of the predicted fruits and vegetables
    availability'''

    df = pd.DataFrame(list(Data.items()), columns=['Year', 'Value'])
# Convert 'Year' column to datetime
    df['Year'] = pd.to_datetime(df['Year'],format='%Y')

# Set 'Year' column as index
    df.set_index('Year', inplace=True)

# Fit ARIMA model
    model = ARIMA(df['Value'], order=(5,1,0))  # Example order, you may need to adjust
    model_fit = model.fit()

    future_dates_past_years = pd.date_range(start=df.index[-1], periods=24, freq='YE') + pd.DateOffset(years=1)
    forecast_past_years = model_fit.forecast(steps=len(future_dates_past_years))

    list_of_past_forecast = forecast_past_years.values
    pastValues = list(Data.values())[-24:]

    absolute_errors = np.abs(pastValues - list_of_past_forecast)

# Calculate mean absolute error (MAE)
    mae = np.mean(absolute_errors)
# Calculate mean squared error (MSE)
    mse = np.mean(absolute_errors ** 2)
# Calculate root mean squared error (RMSE)
    rmse = np.sqrt(mse)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    return absolute_errors, mae, mse, rmse


def appendNewValues(dict, list):
    '''Adds predicted values to the dictionary of availabilty'''

    startYear = 2020
    for i in range(len(list)):
        dict[startYear] = list[i]
        startYear += 1


#While true menu allows for user to select what data they want to see
while True:
    print('Enter F for fruits')
    print('Enter V for vegetables')
    print('Enter FV for fruits and vegetables')
    print('Enter E to exit')
    choice = input('Enter menu selection: ')

    if choice == 'E':
        break

    elif choice == 'V':
        totalVeg = load_data('fruitveg.csv', 750, 799)
        chart, list_of_values = univariate_calculation(totalVeg)
        appendNewValues(totalVeg, list_of_values)
        print(chart)
        absolute_errors, mae, mse, rmse = error_calculation(totalVeg)
        plot_dict(totalVeg)

    elif choice == 'F':
        totalFruits = load_data('fruitveg.csv', 350, 399)
        chart, list_of_values = univariate_calculation(totalFruits)
        appendNewValues(totalFruits, list_of_values)
        print(chart)
        absolute_errors, mae, mse, rmse = error_calculation(totalFruits)
        plot_dict(totalFruits)

    elif choice == 'FV':
        totalFruitsAndVeg = load_data('fruitveg.csv', 800, 852)
        chart, list_of_values = univariate_calculation(totalFruitsAndVeg)
        appendNewValues(totalFruitsAndVeg, list_of_values)
        print(chart)
        absolute_errors, mae, mse, rmse = error_calculation(totalFruitsAndVeg)
        plot_dict(totalFruitsAndVeg)
