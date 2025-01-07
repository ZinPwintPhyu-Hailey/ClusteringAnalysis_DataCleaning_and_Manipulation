#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 19:54:34 2025

@author: zinpwint
"""

# Data Preprocessing
# Step 1: Setup and Initialization 
# Step 1.1: Import necessary library

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans #Import KMeans clustering algorithm from scikit-learn 
from sklearn.preprocessing import StandardScaler # For scaling purpose to standardize the numeric figures 
from sklearn.ensemble import IsolationForest #For Isolation Forest to detect the outliers
from sklearn.metrics import silhouette_score #To check Silhouette score 
from sklearn.decomposition import PCA
from collections import Counter


# Step 1.2: Read the dataset 
# Specify the correct file path

file_path = '/Users/zinpwint/Desktop/Portfolio Projects/Python/Jupyter_Clustering Analysis on Customer Profiles_Kmeans Clustering/AIB503_ECA01_Z2471556_ZinPwintPhyu_04Nov2024/Raw data and python codes/Clustering_data.csv'

# Read the dataset specifying the delimiter as '\t' to handle tab-separated values

data = pd.read_csv(file_path, delimiter='\t', encoding='utf-8')

# Step 2: Initial Data Analysis
# Step 2.1: Data Overview 

print (data.head(5))
print (data.info())

# Step 2.2 Initial Descriptive analysis

descriptive_analysis = data.describe ().T
print (descriptive_analysis)

# Check including 'object' columns 

descriptive_analysis_obj = data.describe (include = 'object')
print (descriptive_analysis_obj)

# Step 3 Data Cleaning and Transformation
# Step 3.1 Checking Duplicates

print (data [data.duplicated(keep=False)])

# Step 3.2 Handling Missing Values
# Count the numbers of missing values in each column

missing_values = data.isnull().sum(axis=0)
print (missing_values)

# Before deciding how to handle the missing value in Income column, 
# I decided to plot the distribution of the 'Income' column to decide which imputation method I should use. 

# Plotting the distribution of the 'Income' column 
plt.figure(figsize=(12,6))
plt.hist(data['Income'].dropna(), bins = 30, color = 'wheat', edgecolor= 'grey')
plt.title('Distribution plot for Income')
plt.xlabel('Income')
plt.ylabel('Frequency') 
plt.grid(False)
plt.show()  

# It is right-skewed plot with most values concentrated between 0 and 100,000. 
# In this cases, using the median to fill the missing values is often preferred 
# because it is less affected by extreme values (outliers) compared to the mean.

# Proceed to imputate the missing values with 'Median' value for Income Column 
median_income = data['Income'].median()

# Replace the 'median' value in the missing values of Income column 
data['Income'].fillna(median_income, inplace = True)

# Check again if there is any remaining missing values in the dataset
data.isnull().sum(axis=0)

# No more missing values, especially in 'Income' column. 

# Step 3.3 Identify the garbage values in 'object' columns 

for i in data.select_dtypes(include="object").columns:
    print (data [i]. value_counts())
    print ("***"*10)

# To Clean the 'Marital_Status' column
# Replace 'Alone', 'Absurd' and 'YOLO' with 'Other'
data['Marital_Status'] = data['Marital_Status'].replace(['Alone','Absurd', 'YOLO'], 'Other')

# Verify the cleaning
print(data['Marital_Status'].value_counts())

# Convert 'Dt_Customer' column to datetime format
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')

# Verify the conversion
print(data['Dt_Customer'].head())

# Step 4: Feature Engineering
# Create a copy of the original data to add the engineered features

data_features = data.copy()

# Step 4.1: Find the customer age 

data_features['Customer_Age'] = datetime.now().year - data_features['Year_Birth']

# Step 4.2: Calculate 'Customer_tenure' 

data_features['Customer_tenure'] = (pd.Timestamp('today') - data_features['Dt_Customer']).dt.days//30

# Step 4.3: Calculate 'Family Size'

data_features['Family_size'] = data_features['Kidhome'] + data_features['Teenhome'] + 2 #Generally assuming 2 adults at home

# Step 4.4: Calculate 'Total Spending in last two years'

data_features['Total_spending'] = data_features[['MntWines', 'MntFruits', 'MntMeatProducts', 
                                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

# Step 4.5: Calculate Average Spending per Purchase

data_features['Total_purchase'] = data_features[['NumDealsPurchases', 'NumWebPurchases', 
                                'NumCatalogPurchases', 'NumStorePurchases']].sum(axis=1)

data_features['Avg_spending/purchase'] = data_features['Total_spending'] / data_features['Total_purchase']

# Step 4.6: Campaign acceptance rate

campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
data_features['Campaign_Acceptance_Rate'] = data_features[campaign_cols].mean(axis=1)

# Step 4.7: Preferred Purchase Channel

data_features['Preferred_Purchase_Channel'] = data_features[['NumWebPurchases', 'NumCatalogPurchases', 
                                                             'NumStorePurchases']].idxmax(axis=1)

# Step 4.8: Complaint Response Interaction

data_features['Complaint_Response_Interaction'] = data_features['Complain'] * data_features['Response']

# Step 5: Outliers Detection and Treatment
# Step 5.1: IQR Outlier Detection and Handling
# Select only numeric columns and exclude the Derived Columns such as Campaign Acceptance Rate

columns = data_features.select_dtypes(include="number").columns
columns = columns.drop(['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4',
                        'AcceptedCmp5', 'Campaign_Acceptance_Rate', 'Complain', 
                        'Z_CostContact', 'Z_Revenue', 'Response'])
    
# Define a function to calculate IQR and apply capping on outliers
def iqr_outlier_handling(data_features, columns):
    # Create a dictionary to store the outlier counts for each column
    outlier_counts = {}

    # Calculate IQR and handle outliers
    for col in columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = data_features[col].quantile(0.25)
        Q3 = data_features[col].quantile(0.75)
        IQR = Q3 - Q1  # Calculate IQR

        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count the outliers before capping for record-keeping
        outliers = data_features[(data_features[col] < lower_bound) | (data_features[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
        
        # Cap the outliers
        data_features[col] = data_features[col].clip(lower=lower_bound, upper=upper_bound)
        
    return data_features, outlier_counts

# Apply the IQR outlier handling to the dataset's numeric columns
data_iqr_handled, outlier_counts = iqr_outlier_handling(data_features, columns)

# Display the number of outliers detected per column
print("Outliers detected per column before capping:")
print(outlier_counts)

# Step 5.2: Outlier treatment 
# Select only numeric columns

numeric_data = data_features.select_dtypes(include=['number'])

# Initializing the IsolationForest model with a contamination parameter of 0.05
model = IsolationForest(contamination=0.05, random_state=0)

# Fitting the model on our dataset (converting DataFrame to NumPy to avoid warning)
data_features['Outlier_Scores'] = model.fit_predict(numeric_data.to_numpy())
# -1 indicates an outlier, 1 indicates an inlier

# Creating a new column to identify outliers 
data_features['Is_Outlier'] = [1 if x == -1 else 0 for x in data_features['Outlier_Scores']]

# Display the first few rows of the data_features dataframe
data_features.head()

# After applying the Isolation Forest algorithm, I have identified the outliers and marked them 
# in a new column named Is_Outlier. I have also calculated the outlier scores which represent the 
# anomaly score of each record.
# Now I will visualize the distribution of these scores and the number of inliers and outliers detected 
# by the model. 

# Calculate the percentage of inliers and outliers
outlier_percentage = data_features['Is_Outlier'].value_counts(normalize=True) * 100

# Plotting the percentage of inliers and outliers
plt.figure(figsize=(12, 4))
outlier_percentage.plot(kind='barh', color='#ff6200')

# Adding the percentage labels on the bars
for index, value in enumerate(outlier_percentage):
    plt.text(value, index, f'{value:.2f}%', fontsize=15)

plt.title('Percentage of Inliers and Outliers', fontsize=14, fontweight='bold', color='darkblue')
plt.xticks(ticks=np.arange(0, 115, 5))
plt.xlabel('Percentage (%)', fontsize=12)
plt.ylabel('Is Outlier', fontsize=12)
plt.grid(False)
plt.gca().invert_yaxis()
plt.show()

#From the above plot, I can observe that about 5% of the customers have been identified as outliers in our dataset. 

# Separate the outliers for analysis
outliers_data = data_features[data_features['Is_Outlier'] == 1]

# Remove the outliers from the main dataset
data_features_cleaned = data_features[data_features['Is_Outlier'] == 0]

# Drop the 'Outlier_Scores' and 'Is_Outlier' columns
data_features_cleaned = data_features_cleaned.drop(columns=['Outlier_Scores', 'Is_Outlier'])

# Reset the index of the cleaned data
data_features_cleaned.reset_index(drop=True, inplace=True)

# Getting the number of rows in the cleaned customer dataset
data_features_cleaned.shape[0]

# Step 6: Data Standardization

# Initialize the StandardScaler 
scaler = StandardScaler()

# Selecting columns to standardize
columns_to_scale = ['Year_Birth','Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                    'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Customer_Age', 
                    'Total_spending','Total_purchase']

# Apply scaling only on the selected columns and create a DataFrame with scaled data
data_scaled = pd.DataFrame(scaler.fit_transform(data_features_cleaned[columns_to_scale]), columns=columns_to_scale)

# Display the first few rows and info of the scaled data
print(data_scaled.head())
print(data_scaled.info())

print(data_scaled.mean())  # Should be close to 0
print(data_scaled.std())   # Should be close to 1