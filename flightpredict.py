#This is the first part of the project 😀.
"""In day 1 we will simply import some basic library and will perform various functions on our imported data set."""
"""The data set which we will be using is taken from Kaggle and it's an airplane tickets data"""
"""LET'S GET STARTED !!! 🤝"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_excel(r'C:\Users\91766\Downloads\Data_Train.xlsx')
print(data.head())
print(data.info())
print(data.shape)
print(data.count())
print(data.dtypes)
print(data.describe())
print(data.isna().sum())
print(data[data['Route'].isna() | data['Total_Stops'].isna()])
data.dropna(inplace= True)
print(data.isna().sum())
print(data.count())


#This is the Day 2 of our project.
""" Now for Exploratory Data Analysis & Feature Engineering we will explore : """
#1. Duration
#2. Deparature and Arrival Time
#3.Date of journey
#4. Total stops
#5. Additional Info
#6. Airline
#7. Source and Destination
#8. Route
"""LET'S GET STARTED !!!"""

#DURATION
def convert_duration(duration):
    if len(duration.split()) == 2 :
         hours = int(duration.split()[0][:-1])
         minutes = int(duration.split()[1][:-1])
         return hours * 60 +minutes
    else:
        return int(duration[:-1]) * 60
    
data['Duration'] = data['Duration'].apply(convert_duration)
print(data.head())

#DEPARTURE AND ARRIVAL TIME
data['Dep_Time'] = pd.to_datetime(data['Dep_Time'])
data['Arrival_Time'] = pd.to_datetime(data['Arrival_Time'])
print(data.dtypes)
data['Dep_Time_in_hours'] = data['Dep_Time'].dt.hour
data['Dep_Time_in_minutes'] = data['Dep_Time'].dt.minute
data['Arrival_Time_in_hours'] = data['Arrival_Time'].dt.hour
data['Arrival_Time_in_minutes'] = data['Arrival_Time'].dt.minute
print(data.head())
data.drop(['Dep_Time','Arrival_Time'],axis = 1,inplace = True)
print(data.head())

#DATE OF JOURNEY 
data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], dayfirst=True)
print(data.head())
print(data['Date_of_Journey'].dt.year.unique())
data['Day'] = data['Date_of_Journey'].dt.day
data['Month'] = data['Date_of_Journey'].dt.month
print(data.head())
data.drop('Date_of_Journey',axis = 1,inplace =True)
print(data.head())

#TOTAL STOPS
print(data['Total_Stops'].value_counts())
data['Total_Stops'] = data['Total_Stops'].map({
    'non-stop' :0,
    '1 stop' : 1,
    '2 stop' : 2,
    '3 stop' : 3,
    '4 stop' : 4,
})
print(data.head())

#ADDITIONAL INFO
print(data['Additional_Info'].value_counts())
data.drop('Additional_Info',axis=1,inplace=True)
print(data.head())
print(data.select_dtypes(['object']).columns)
for i in ['Airline', 'Source', 'Destination', 'Route']:
    plt.figure(figsize=(15,6))
    sns.countplot(data=data,x=i)
    ax = sns.countplot(x=i,data=data.sort_values('Price',ascending=True))
    ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha ='right')
    plt.tight_layout()
    plt.show()
    print('\n\n')

#AIRLINE
print(data['Airline'].value_counts())
plt.figure(figsize=(15,6))
ax = sns.barplot(x='Airline',y='Price',data=data.sort_values('Price',ascending= False))
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha ='right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,6))
ax = sns.boxplot(x='Airline',y='Price',data=data.sort_values('Price',ascending= False))
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
plt.tight_layout()
plt.show()

print(data.groupby('Airline').describe()['Price'].sort_values('mean',ascending=False))

'''ENCODING :'''
Airline = pd.get_dummies(data['Airline'],drop_first=True)
print((Airline.head()))

data = pd.concat([data,Airline],axis=1)
print(data.head())
 
data.drop('Airline',axis=1,inplace=True)
print(data.head())




#SOURCE AND DESTINATION
list1 = ['Source', 'Destination']
for i in list1:
    print(data[i].value_counts(), '\n')
    
'''Encoding'''
data = pd.get_dummies(data=data,columns=list1,drop_first=True)
print(data.head())



#ROUTE
from sklearn.preprocessing import LabelEncoder

# Create a new DataFrame with the 'Route' column
route = data[['Route']].copy()

# Split the 'Route' column into separate columns
split_routes = route['Route'].str.split('→', expand=True)

# Assign the split values to new columns in the 'Route' DataFrame
route[['Route_1', 'Route_2', 'Route_3', 'Route_4', 'Route_5']] = split_routes.iloc[:, :5]

# Display the first few rows of the 'Route' DataFrame after splitting
print(route.head())

# Fill missing values with 'None' in the 'Route' DataFrame
route.fillna('None', inplace=True)

# Display the first few rows of the 'Route' DataFrame after filling missing values
print(route.head())

# Apply LabelEncoder to each 'Route' column
for i in range(1, 6):
    col = 'Route_' + str(i)
    le = LabelEncoder()
    route[col] = le.fit_transform(route[col])

# Drop the original 'Route' column
route.drop('Route', axis=1, inplace=True)

# Display the first few rows of the 'Route' DataFrame after LabelEncoding
print(route.head())

# Concatenate 'route' DataFrame with the original 'data' DataFrame
data = pd.concat([data, route], axis=1)

# Drop the original 'Route' column from the 'data' DataFrame
data.drop('Route', axis=1, inplace=True)

# Display the first few rows of the updated 'data' DataFrame
print(data.head())
