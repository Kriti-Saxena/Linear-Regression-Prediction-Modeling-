# Linear Regression Modeling for Predictions 

# **Data Cleaning and Preprocessing for Linear Regression Modeling**

This notebook will primarily focus on cleaning and preprocessing a data set for linear regession modeling.
The notebook uses dataset that contains data of used cars with other associated variables such as price it was purchased for, the mileage, model, year it was manufactured, engineV, engine type, and body.
The aim is to predict reselling price pof the cars on the basis of these specifications.

# Import required libraries
```
import numpy as np
import pandas as pd 
import seaborn as sn 
import statsmodels.api as sma
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import math as m
```

# loading data 
```
data = pd.read_csv('Dataset.csv')
```

# exploring the dataset 
```
data.describe()
```

**Identifying Variables**
Brand is an important factor, the more luxury the brand is, the more expensive is the car.
Mileage - The more the car has already been used, the cheaper it would be
Year is also important. Typically, the older the car the cheaper, however, this is not true in the case because there are some vinatge cars as well
Engine is also identifed as one factor to be considered.
These are some of many variables.
for this model we will not consider model as one of the predictor valiable consider 

```
data = data.drop(["Model"],axis=1)
```

# Checking for null values 
```
data.isnull().sum()
```
# Structuring and cleaning the data 

##### Dropping Missing values because they are less than 5% of the whole dataset 
```
data = data.dropna(axis=0)
```

# checking for skewness and outliers of the data 
```
sn.displot(data['Price'])
```
It is evident from the graph that there are some outliers in price variable. The outliers are towards the higher price end i.e the right side of the graph. To deal with these outliers we can consider the data that belongs under the 99th percentile.

```
Q = data['Price'].quantile(0.99)
data = data[data['Price']<Q]
data.describe(include='all')
```
# Checking again 
```
sn.displot(data['Price'])
```
The number of outliers have reduced. 

##### Checking for outliers in different variables and dealing with them 
```
sn.displot(data['Mileage'])
q = data['Mileage'].quantile(0.99)
data = data[data['Mileage']<q]
```
```
sn.displot(data['EngineV'])
data = data[data['EngineV']<6.5]
sn.displot(data['EngineV'])
```
```
sn.displot(data["Year"])
```
The outliers in this case is on the left side of the data, therefore we will only take the data that is more than 1% of the whole dataset, essentially take the outliers on left side out 

```
o = data['Year'].quantile(0.01)
data = data[data['Year']>o]
sn.displot(data['Year'])
```

```
data = data.reset_index(drop=True)
data.describe(include='all')
```

# Preprocessing Data for Linear Regression Modeling 
```
# checking OLS Assumptions 
# checking for linearity of the data 

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) 
ax1.scatter(data['Year'],data['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data['EngineV'],data['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data['Mileage'],data['Price'])
ax3.set_title('Price and Mileage')


plt.show()
```
None of these have linear realtionship, therefore we will, transform one of the variable. Lets look at the price variable again 

```
sn.displot(data["Price"])
log_price = np.log(data['Price'])
data['log_price'] = log_price
data = data.drop(["Price"],axis=1) # because we will only use log_price variable
```

```
# checking the data again using graph 

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) 
ax1.scatter(data['Year'],data['log_price'])
ax1.set_title(' Log Price and Year')
ax2.scatter(data['EngineV'],data['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data['Mileage'],data['log_price'])
ax3.set_title('Log Price and Mileage')


plt.show()
```

Now, a linear pattern can be seen in the graph 

#### Checking for multicollinearity 

We will use variance inflation factor to test the variables for multicollinearity 

```
data.columns.values
```

```
variables = data[['Mileage','Year','EngineV']]
vif = pd.DataFrame()


vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

vif["Features"] = variables.columns
vif
```
Drop the year variable because it has highest vif value, this will also potentially drive down the vif value for other variables 

#### Creating Dummy Variables 

```
## Creating dummies for catagorical variable 

data = pd.get_dummies(data, drop_first = True)
```

```
# Rearranging the data
data.columns.values
cols = ['log_price','Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
data = data[cols]
data
```

## Linear Regression Modeling 

```
target = data['log_price']
inputs = data.drop(['log_price'], axis = 1)

# scaling the data to normalize it 

scaler = StandardScaler()
scaler.fit(inputs)
```
```
inputs_s = scaler.transform(inputs)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(inputs_s, target, test_size=0.2, random_state=365)
# creating regression 
reg = LinearRegression()
```
```
# fitting the model 
reg.fit(x_train, y_train)
y_pred = reg.predict(x_train)
# comparing the target and predictions 

plt.scatter(y_train, y_pred)
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
```
```
# finding R-Squared value
reg.score(x_train, y_train)
reg.intercept_
reg.coef_
```
Model explains Approximately 74.49% variability of data 

```
# Creating Summary Table 
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary
```

# Model Evaluation 
```
plt.scatter(y_train, y_pred, alpha = 0.1)
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
```
From the graph it is seen that overall, our model is good at predicting prices of cars, but when seen properly, it can be noticed that our model is better at predicting prices for cars of higher value than for lower value. As seen, the points are scattered and less visible at the end, indicating that predictions are off the target and the density of points on the right side of the graph is better.

```
# looking at residual plots 
sn.displot(y_train - y_pred)
```
The graph is skewed to right, the long tail with values below mean indicates that the model is overestimating the values, but the short tail on the right side indicates that the model rearly underestimates the values 

```
# checking the prices 
prices = pd.DataFrame(np.exp(y_pred),columns = ['Prediction'])
prices['Target'] = np.exp(y_test)
```

# Indexing for y_test 
```
y_test = y_test.reset_index(drop = True)
prices['Target'] = np.exp(y_test) # overriding 
prices
```
```
prices['Difference'] = prices['Target'] - prices ['Prediction']
pd.options.display.max_rows = 999
prices
```
```
sn.displot(prices['Difference'])
```

# % differnece 
```
prices['Difference%'] = np.absolute(prices['Difference']/prices['Target']*100)
prices.head()
```
```
sn.displot(prices['Difference%'])
```

# Takeaways 

From the qunatiles it can be seen that the predictions made by the model were fairly close. The minimum difference is 0.2% However, the maximum difference is too large. Indicating that the models predictions were a bit off. Furthermore, mosr of the difference is in negative vvalues, it can mean many thing, such as the difference is also because some of the other variables which were not taken into consideration like the Model of the car. This is Certainly not the best model, however, there are ways to make it better.
