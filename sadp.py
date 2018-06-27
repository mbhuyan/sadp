#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:52:32 2018

@author: Pratyush Kumar Deka
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
# from glob import glob
# from scipy.stats.stats import pearsonr
# import seaborn as sns

"""
# Convert index to datetime
start = pd.to_datetime(1416726000, unit='s')
end = pd.to_datetime(1421838000, unit='s')
index = pd.date_range(start, end, freq='H')

"""

'''
Load data and get all the possible information
'''
real_1 = pd.read_csv("Webscope_S5/A1Benchmark/real_1.csv")
print("real_1.csv file")
real_1.info()
print("real_1.csv - First 5 lines")
print(real_1.head())
print("real_1.csv - Last 5 lines")
print(real_1.tail())

synthetic_1 = pd.read_csv("Webscope_S5/A2Benchmark/synthetic_1.csv")
print("synthetic_1.csv file")
synthetic_1.info()
print("synthetic_1.csv - First 5 lines")
print(synthetic_1.head())
print("synthetic_1.csv - Last 5 lines")
print(synthetic_1.tail())


'''
Convert to datetime if required, and set the index to timestamp
'''

real_1 = real_1.set_index('timestamp')
print("real_1.csv - After timestamp conversion")
real_1.info()
print("real_1.csv - First 5 lines after timestamp conversion")
print(real_1.head())

synthetic_1['timestamp'] = pd.to_datetime(synthetic_1['timestamp'], unit='s')
print("AFTER TIMESTAMP CONVERSION")
synthetic_1 = synthetic_1.set_index('timestamp')
print("synthetic_1.csv - After timestamp conversion")
synthetic_1.info()
print("synthetic_1.csv - First 5 lines after timestamp conversion")
print(synthetic_1.head())

''' Check whether any Null value is present '''

print("real_1.csv - Check for any NULL value - " + str(real_1.isnull().sum()))
print("synthetic_1.csv - Check for any NULL value - " + str(synthetic_1.isnull().sum()))


'''Graph plot'''

plt.style.use('ggplot')

rx = real_1.plot(figsize=(14,6), linewidth=0.8, fontsize=6)
rx.set_xlabel('Timestamp')
rx.set_ylabel('Value')
rx.set_title('real_1.csv')
plt.show()

sx = synthetic_1.plot(figsize=(20,8), linewidth=0.8, fontsize=6)
sx.set_xlabel('Timestamp')
sx.set_ylabel('Value')
sx.set_title('synthetic_1.csv')
plt.show()

real_1.plot(subplots=True, figsize=(20,10), layout=(2,1), fontsize=8, linewidth=0.5)
plt.show()

synthetic_1.plot(subplots=True, figsize=(20, 10), layout=(2,1), fontsize=8, linewidth=0.5)
plt.show()

''' SUMMARY STATISTICS '''
print("real_1.csv - Summary statistics")
print(real_1.describe())

print("synthetic_1.csv - Summary statistics")
print(synthetic_1.describe())

''' Display the autocorrelation plot of time series '''
print("Autocorrelation of real_1.csv")
real_1_corr = tsaplots.plot_acf(real_1['value'], lags=24)
plt.show()

print("Autocorrelation of synthetic_1.csv")
synthetic_1_corr = tsaplots.plot_acf(synthetic_1['value'], lags=24)
plt.show()


synthetic_1_decomposition = sm.tsa.seasonal_decompose(synthetic_1['value'])
# Extract the trend component
trend = synthetic_1_decomposition.trend
seasonal =  synthetic_1_decomposition.seasonal
residual = synthetic_1_decomposition.resid
# Plot the values of the trend
tx = trend.plot(figsize=(12, 6), fontsize=6, linewidth=.6)
# Specify axis labels
tx.set_xlabel('Time', fontsize=10)
tx.set_title('Trend', fontsize=10)
plt.show()

sx = seasonal.plot(figsize=(12, 6), fontsize=6, linewidth=.6)
tx.set_xlabel('Time', fontsize=10)
tx.set_title('Seasonal', fontsize=10)
plt.show()

rx = residual.plot(figsize=(12, 6), fontsize=6, linewidth=.6)
# Specify axis labels
tx.set_xlabel('Time', fontsize=10)
tx.set_title('Residual', fontsize=10)
plt.show()

ax = synthetic_1_decomposition.plot()
plt.show()
