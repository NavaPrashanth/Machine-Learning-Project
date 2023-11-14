#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset
file_path = 'energydata_complete.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())


# In[2]:


# Get a quick description of the data
print(data.describe())


# In[3]:


# Check for missing values
print("Missing values in each column:\n", data.isnull().sum())


# In[4]:


# Check the data types of each column
print("Data types:\n", data.dtypes)


# In[5]:


# Assuming your date column is named "date_column"
data['date'] = pd.to_datetime(data['date'])


# In[6]:


# Setting date as the index:
data.set_index('date', inplace=True)


# In[7]:


# Dataset Duplicate Value Count assinged a dataframe name 'df'
df = data[data.duplicated()]

#There is no duplicate rows in the data
df.head()

# Missing Values/Null Values Count
data.isna().sum()


# In[8]:


# Dataset Columns
data.columns


# In[9]:


# Basic statistics
print("\nBasic statistical details:\n", data.describe())


# In[10]:


data.head()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the correlation matrix
corr_matrix = data.corr()

# Plotting the heatmap
plt.figure(figsize=(15, 15))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()


# In[12]:


# Identifying feature groups by their names:
temp_features = [col for col in data.columns if "T" in col]
humid_features = [col for col in data.columns if "RH" in col]
other_columns = [col for col in data.columns if not ("T" in col or "RH" in col)]


# In[13]:


#close look on temprature column
data[temp_features].describe(include='all')


# In[14]:


#close look on temprature column
data[humid_features].describe(include='all')


# In[15]:


#close look on temprature column
data[other_columns].describe(include='all')


# In[16]:


# Mapping to rename temperature and humidity columns to more descriptive names
rename_columns = {
    'T1': 'TEMP_KITCHEN',
    'RH_1': 'HUM_KITCHEN',
    'T2': 'TEMP_LIVING_ROOM',
    'RH_2': 'HUM_LIVING_ROOM',
    'T3': 'TEMP_BEDROOM',
    'RH_3': 'HUM_BEDROOM',
    'T4': 'TEMP_OFFICE',
    'RH_4': 'HUM_OFFICE',
    'T5': 'TEMP_BATHROOM',
    'RH_5': 'HUM_BATHROOM',
    'T6': 'TEMP_OUTSIDE_BUILDING',
    'RH_6': 'HUM_OUTSIDE_BUILDING',
    'T7': 'TEMP_IRONING_ROOM',
    'RH_7': 'HUM_IRONING_ROOM',
    'T8': 'TEMP_TEEN_ROOM_2',
    'RH_8': 'HUM_TEEN_ROOM_2',
    'T9': 'TEMP_PARENTS_ROOM',
    'RH_9': 'HUM_PARENTS_ROOM',
    'T_out': 'TEMP_OUTSIDE_WEATHER_STATION',
    'RH_out': 'HUM_OUTSIDE_WEATHER_STATION'
}


# In[17]:


# Applying the column name changes to the DataFrame
data.rename(columns=rename_columns, inplace=True)


# In[18]:


data.head()


# In[19]:


import pandas as pd

# Assuming 'data' is your DataFrame with a DatetimeIndex
data['month'] = data.index.month
data['weekday'] = data.index.weekday
data['time_of_day'] = data.index.hour
data['week_number'] = data.index.strftime('%U').astype(int) + 1  # %U gives the week number, add 1 to start from 1
data['date_day'] = data.index.day
data['weekday_name'] = data.index.dayofweek


# In[20]:


data.head(2)


# In[21]:


# Analyzing the distribution of 'lights' column values
light_distribution = data['lights'].value_counts(normalize=True) * 100
light_distribution


# In[22]:


# Excluding the 'lights' column from the dataset
data.drop('lights', axis=1, inplace=True)


# In[23]:


# Adjusting the order of DataFrame columns for better readability
sorted_columns = [
    'TEMP_KITCHEN', 'TEMP_LIVING_ROOM', 'TEMP_BEDROOM', 'TEMP_OFFICE', 'TEMP_BATHROOM', 
    'TEMP_OUTSIDE_BUILDING', 'TEMP_IRONING_ROOM', 'TEMP_TEEN_ROOM_2', 'TEMP_PARENTS_ROOM', 
    'TEMP_OUTSIDE_WEATHER_STATION', 'HUM_KITCHEN', 'HUM_LIVING_ROOM', 'HUM_BEDROOM', 'HUM_OFFICE', 
    'HUM_BATHROOM', 'HUM_OUTSIDE_BUILDING', 'HUM_IRONING_ROOM', 'HUM_TEEN_ROOM_2', 'HUM_PARENTS_ROOM', 
    'HUM_OUTSIDE_WEATHER_STATION', 'Tdewpoint', 'Press_mm_hg', 'Windspeed', 'Visibility', 'rv1', 'rv2', 
    'month', 'weekday', 'time_of_day', 'week_number', 'date_day', 'weekday_name', 'Appliances'
]
data = data.reindex(columns=sorted_columns)


# In[24]:


data.head(2)


# In[25]:


# Visualizing average daily appliance energy consumption
daily_avg_energy = data.pivot_table(values='Appliances', index='date_day', columns='month', aggfunc='mean')

# Generating a heatmap for daily energy use
plt.figure(figsize=(12, 7))
sns.heatmap(daily_avg_energy, cmap='viridis')
plt.title('Average Daily Appliance Energy Use')
plt.xlabel('Month')
plt.ylabel('Day of the Month')
plt.show()


# In[26]:


# Check unique values in the 'weekday_name' column
unique_weekdays = data['weekday_name'].unique()

# Create a box plot or violin plot to compare energy consumption across different days of the week
plt.figure(figsize=(10, 6))
sns.boxplot(x='weekday_name', y='Appliances', data=data, order=unique_weekdays)  # or sns.violinplot()
plt.title('Appliance Energy Consumption by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Energy Consumption')


# In[27]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Visualizing the distribution of values in the 'Appliances', 'Windspeed', 'Visibility', and 'Press_mm_hg' columns
fig_sub = make_subplots(rows=1, cols=4, subplot_titles=('Appliances Distribution', 'Windspeed Distribution', 
                                                        'Visibility Distribution', 'Pressure Distribution'))

fig_sub.add_trace(go.Box(y=data['Appliances'], name='Appliances'), row=1, col=1)
fig_sub.add_trace(go.Box(y=data['Windspeed'], name='Windspeed'), row=1, col=2)
fig_sub.add_trace(go.Box(y=data['Visibility'], name='Visibility'), row=1, col=3)
fig_sub.add_trace(go.Box(y=data['Press_mm_hg'], name='Pressure'), row=1, col=4)

fig_sub.update_layout(height=600, width=800, title_text="Box Plots of Selected Variables")
fig_sub.show()


# In[28]:


# Ensure the 'date' column is in datetime format and set it as the index if not already done
if not isinstance(data.index, pd.DatetimeIndex):
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

# Extract 'hour' and 'weekday' from the datetime index
data['hour'] = data.index.hour
data['weekday'] = data.index.weekday

# Now proceed with the grouping and plotting
weekday_consumption = data[data['weekday'] < 5].groupby('hour')['Appliances'].mean()
weekend_consumption = data[data['weekday'] >= 5].groupby('hour')['Appliances'].mean()

plt.figure(figsize=(12, 7))
plt.plot(weekday_consumption.index, weekday_consumption.values, label='Weekdays', marker='o')
plt.plot(weekend_consumption.index, weekend_consumption.values, label='Weekends', marker='o')
plt.title('Energy Consumption on Weekdays vs. Weekends')
plt.xlabel('Hour of the Day')
plt.ylabel('Mean Energy Consumption')
plt.xticks(range(24))
plt.grid(True)
plt.legend()
plt.show()


# In[29]:


data.columns


# In[30]:


# Hypothesis testing for correlation between features and energy consumption
from scipy.stats import pearsonr

# Independent and dependent variables separation
independent_vars = data.drop(['Appliances'], axis=1)
dependent_var = data['Appliances']

# Convert all columns to numeric, coerce errors for non-numeric data
independent_vars = independent_vars.apply(pd.to_numeric, errors='coerce').fillna(0)

# Ensure equal length
length = min(len(independent_vars), len(dependent_var))
independent_vars = independent_vars.iloc[:length]
dependent_var = dependent_var.iloc[:length]

# Pearson correlation test for each feature
correlation_results = [(feature, *pearsonr(independent_vars[feature], dependent_var)) 
                       for feature in independent_vars.columns]

# Output the correlation results with significance testing
for feature, corr_coefficient, p_value in correlation_results:
    print(f"Correlation for {feature} is {corr_coefficient:.4f} with a p-value of {p_value:.4g}")
    if p_value < 0.05:
        print(f"The correlation between {feature} and appliance energy consumption is statistically significant.")
    else:
        print(f"No significant correlation found for {feature}.")
    print()  # For better readability


# In[31]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame
df = data.copy()
col_list = list(df.describe().columns)

# Calculate the number of rows needed for subplots
num_features = len(col_list)
num_columns = 4  # You can keep the number of columns as 4 or change it as needed
num_rows = (num_features + num_columns - 1) // num_columns  # This will round up the division

# Create the box plots
plt.figure(figsize=(25, num_rows * 5))  # Adjust the size based on the number of rows
plt.suptitle("Box Plot", fontsize=18, y=0.95)

for n, ticker in enumerate(col_list):
    ax = plt.subplot(num_rows, num_columns, n + 1)
    sns.boxplot(x=df[ticker], color='cyan')
    ax.set_title(ticker.upper())
    plt.subplots_adjust(hspace=0.5, wspace=0.2)  # Adjust the spacing if needed

plt.show()


# In[32]:


import pandas as pd
import numpy as np

def find_outliers_iqr(data):
    # Convert data to numeric (ignore errors to handle non-numeric values)
    data_numeric = data.apply(pd.to_numeric, errors='coerce')

    # Calculate the first quartile (Q1) and third quartile (Q3) for each column
    q1 = data_numeric.quantile(0.25)
    q3 = data_numeric.quantile(0.75)

    # Calculate the interquartile range (IQR) for each column
    iqr = q3 - q1

    # Calculate the lower and upper bounds for outliers for each column
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Check for outliers in each column and count the number of outliers
    outliers_count = (data_numeric < lower_bound) | (data_numeric > upper_bound)
    num_outliers = outliers_count.sum()

    return num_outliers

outliers_per_column = find_outliers_iqr(data)
print("Number of outliers per column:")
print(outliers_per_column.sort_values(ascending=False))


# In[33]:


# Handling Outliers & Outlier treatments
for ftr in col_list:
  print(ftr,'\n')
  q_25= np.percentile(df[ftr], 25)
  q_75 = np.percentile(df[ftr], 75)
  iqr = q_75 - q_25
  print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q_25, q_75, iqr))
  # calculate the outlier cutoff
  cut_off = iqr * 1.5
  lower = q_25 - cut_off
  upper = q_75 + cut_off
  print(f"\nlower = {lower} and upper = {upper} \n ")
  # identify outliers
  outliers = [x for x in df[ftr] if x < lower or x > upper]
  print('Identified outliers: %d' % len(outliers))
  #removing outliers
  if len(outliers)!=0:

    def bin(row):
      if row[ftr]> upper:
        return upper
      if row[ftr] < lower:
        return lower
      else:
        return row[ftr]



    data[ftr] =  df.apply (lambda row: bin(row), axis=1)
    print(f"{ftr} Outliers Removed")
  print("\n-------\n")


# In[34]:


#examining the shape after
data.shape


# In[35]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame
df = data.copy()
col_list = list(df.describe().columns)

# Calculate the number of rows needed for subplots
num_features = len(col_list)
num_columns = 4  # You can keep the number of columns as 4 or change it as needed
num_rows = (num_features + num_columns - 1) // num_columns  # This will round up the division

# Create the box plots
plt.figure(figsize=(25, num_rows * 5))  # Adjust the size based on the number of rows
plt.suptitle("Box Plot", fontsize=18, y=0.95)

for n, ticker in enumerate(col_list):
    ax = plt.subplot(num_rows, num_columns, n + 1)
    sns.boxplot(x=df[ticker], color='cyan')
    ax.set_title(ticker.upper())
    plt.subplots_adjust(hspace=0.5, wspace=0.2)  # Adjust the spacing if needed

plt.show()


# In[36]:


# Use the correct column names from your DataFrame to create new features
data['Avg_Temp_Building'] = data[['TEMP_KITCHEN', 'TEMP_LIVING_ROOM', 'TEMP_BEDROOM', 'TEMP_OFFICE',
                                  'TEMP_BATHROOM', 'TEMP_IRONING_ROOM', 'TEMP_TEEN_ROOM_2', 
                                  'TEMP_PARENTS_ROOM']].mean(axis=1)

data['Delta_Temp_Inside_Outside'] = abs(data['Avg_Temp_Building'] - data['TEMP_OUTSIDE_BUILDING'])

data['Avg_Hum_Building'] = data[['HUM_KITCHEN', 'HUM_LIVING_ROOM', 'HUM_BEDROOM', 'HUM_OFFICE',
                                 'HUM_BATHROOM', 'HUM_IRONING_ROOM', 'HUM_TEEN_ROOM_2', 
                                 'HUM_PARENTS_ROOM']].mean(axis=1)

data['Delta_Hum_Inside_Outside'] = abs(data['Avg_Hum_Building'] - data['HUM_OUTSIDE_BUILDING'])


# In[37]:


# Drop the random variables as they are not important for the prediction
data.drop(['rv1', 'rv2'], axis=1, inplace=True)


# In[38]:


data.shape


# In[39]:


import numpy as np

# Exclude non-numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Calculate skewness for numeric columns
skewness = numeric_data.skew()

# Find the absolute value
abs_skewness = abs(skewness)

# Set up the threshold
skewness_threshold = 0.5

# Separate features into symmetrical and skewed based on skewness threshold
symmetrical_features = abs_skewness[abs_skewness < skewness_threshold].index
skewed_features = abs_skewness[abs_skewness >= skewness_threshold].index

# Create new DataFrames for symmetrical and skewed features
print('FEATURES FOLLOWED SYMMETRICAL DISTRIBUTION:')
symmetrical_data = data[symmetrical_features]
print(symmetrical_features)

print('FEATURES FOLLOWED SKEWED DISTRIBUTION:')
skewed_data = data[skewed_features]
print(skewed_features)


# In[40]:


#examining the skewed data
skewed_data


# In[41]:


#import the liabrary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

# Initialize the PowerTransformer
power_transformer = PowerTransformer()

# Fit and transform the data using the PowerTransformer
power_transformed = pd.DataFrame(power_transformer.fit_transform(skewed_data))
power_transformed.columns = skewed_data.columns


# In[42]:


#examining the power transformed data
power_transformed


# In[43]:


# Reset the index to the default integer index
symmetrical_data.reset_index(drop=True, inplace=True)


# In[44]:


#examining the symmetrical data
symmetrical_data


# In[45]:


# Concatenate horizontally (along columns)
tranformed_data = pd.concat([symmetrical_data, power_transformed], axis=1)


# In[46]:


#examining the transformed data
tranformed_data


# In[47]:


#importing the desired liabrary
from sklearn.preprocessing import StandardScaler

# StandardScaler
scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(tranformed_data))
scaled_data.columns = tranformed_data.columns
scaled_data


# In[48]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Setting up PCA, without initially limiting the number of components
pca_full = PCA()

# Applying PCA to the feature-scaled dataset
pca_full.fit(scaled_data)

# Summing the explained variance ratios to determine how many components to consider
variance_ratios_cumulative = np.cumsum(pca_full.explained_variance_ratio_)

# Plotting the cumulative variance explained by each additional principal component
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(variance_ratios_cumulative) + 1), variance_ratios_cumulative, marker='o', linestyle='--')
plt.xlabel('Count of Principal Components')
plt.ylabel('Total Explained Variance')
plt.title('Cumulative Variance Explained by PCA Components')
plt.grid(True)
plt.show()

# Determining the number of principal components for dimensionality reduction
# For example, to keep 10 components, we set it to 10
num_components_kept = 10
pca_reduced = PCA(n_components=num_components_kept)

# Fitting and transforming the data to reduce its dimensionality
pca_transformed_features = pca_reduced.fit_transform(scaled_data)

# Inspecting the variance explained by each of the selected components
component_variance = pca_reduced.explained_variance_ratio_

# Calculate the sum of explained variance by the components selected
total_explained_variance = np.sum(component_variance)
print(f"Total Explained Variance by {num_components_kept} components: {total_explained_variance:.2f}")


# In[49]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Initializing PCA without pre-setting the number of components to extract
pca_analysis = PCA()

# Conducting PCA on the scaled dataset
pca_analysis.fit(scaled_data)

# Retrieving the variance explained by each principal component
variance_by_component = pca_analysis.explained_variance_ratio_

# Constructing a scree plot to display the variance explained by each component
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(variance_by_component) + 1), variance_by_component, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Variance Explained')
plt.title('Variance Explained by PCA Components')
plt.grid(True)
plt.show()


# In[50]:


pca_transformed_features.shape


# In[51]:


pca_transformed_features


# In[52]:


#assinign the independent and dependent feature
X = pca_transformed_features
y = data['Appliances']


# In[53]:


#splitting the data into 80/20 ration
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=85)


# In[54]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Initialize the Linear Regression estimator
linear_regressor = LinearRegression()

# Fit the model to the training data
linear_regressor.fit(x_train, y_train)

# Evaluate the model on the training set
train_r2 = linear_regressor.score(x_train, y_train)
print("Training R^2 score:", train_r2)

# Predict the target on the testing set
predicted_y = linear_regressor.predict(x_test)

# Calculate and print the Mean Squared Error (MSE) on the test set
test_mse = mean_squared_error(y_test, predicted_y)
print("Test Mean Squared Error (MSE):", test_mse)

# Calculate and print the R-squared score on the test set
test_r2 = r2_score(y_test, predicted_y)
print("Test R^2 score:", test_r2)

# Calculate and print the Mean Absolute Error (MAE) on the test set
test_mae = mean_absolute_error(y_test, predicted_y)
print("Test Mean Absolute Error (MAE):", test_mae)

# If additional metrics are of interest, they could also be calculated here
# For example, you might consider the Root Mean Squared Error (RMSE)
test_rmse = np.sqrt(test_mse)
print("Test Root Mean Squared Error (RMSE):", test_rmse)

# The coefficients from the linear model can provide insight into the importance of each feature
print("Coefficients of the linear model:", linear_regressor.coef_)


# Training R² score: 0.8141 indicates that the model explains 81.41% of the variance in the training data, which is a high score and suggests that the model fits the training data well.
# 
# Test R² score: 0.8201 is slightly higher than the training R² score, which is a positive sign. It means the model explains 82.01% of the variance in the test data and indicates that the model generalizes well to unseen data.
# 
# Test MSE: 344.53 is the average of the squares of the errors between the predicted and actual values in the test set. The lower the MSE, the better the model.
# 
# Test RMSE: 18.56 is the square root of the MSE and provides an error term in the same unit as the target variable, making it more interpretable. This value represents the standard deviation of the residuals (prediction errors).
# 
# Test MAE: 14.62 is the mean of the absolute values of the errors. It provides an idea of how big the average error is in the same unit as the target variable.

# In[55]:


from sklearn.model_selection import cross_val_score

# Define the linear regression model
linear_regressor = LinearRegression()

# Perform k-fold cross-validation
k = 5  # Number of folds
cv_scores = cross_val_score(linear_regressor, X, y, cv=k, scoring='r2')

# Print the cross-validation R² scores and the mean score
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean cross-validation R²: {cv_scores.mean()}")

# The mean cross-validation score provides an estimate of the model's out-of-sample performance.


# In[56]:


import matplotlib.pyplot as plt

# Plot actual vs predicted values
plt.figure(figsize=(10, 8))
plt.scatter(y_test, predicted_y, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', lw=3, color='red')  # Diagonal line representing perfect predictions
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()


# In[57]:


import matplotlib.pyplot as plt

# Plotting Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicted_y, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', lw=2, color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()

# Plotting Residuals
residuals = y_test - predicted_y
plt.figure(figsize=(10, 6))
plt.scatter(predicted_y, residuals, alpha=0.5)
plt.hlines(y=0, xmin=predicted_y.min(), xmax=predicted_y.max(), colors='red', linestyle='--', lw=2)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.show()


# ### XGBoost Algorithm

# In[58]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the XGBoost regressor with specific hyperparameters
xgb_regressor = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, subsample=1.0, random_state=85)

# Fit the regressor to the polynomially transformed training data
xgb_regressor.fit(x_train, y_train)

# Predicting the energy usage for the training and test sets
y_pred_train = xgb_regressor.predict(x_train)
y_pred_test = xgb_regressor.predict(x_test)

# Assessing the regressor's accuracy
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

# Displaying the performance metrics
print("Performance of XGBoost Regressor:")
print(f"Training MSE: {mse_train:.2f}")
print(f"Test MSE: {mse_test:.2f}")
print(f"Training R^2: {r2_train:.4f}")
print(f"Test R^2: {r2_test:.4f}")


# ### Cross validation for XGBoost

# In[59]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_r2_scores = cross_val_score(xgb_regressor, X, y, cv=5, scoring='r2')
cv_mse_scores = cross_val_score(xgb_regressor, X, y, cv=5, scoring='neg_mean_squared_error')

print("Cross-Validation R2 scores:", cv_r2_scores)
print("Mean CV R2:", cv_r2_scores.mean())
print("Cross-Validation MSE scores:", -cv_mse_scores)
print("Mean CV MSE:", -cv_mse_scores.mean())


# ### Hyper Parameter Tuning for XGBoost

# In[60]:


from sklearn.model_selection import GridSearchCV

# Define a new parameter grid based on the previous results
param_grid = {
    'max_depth': [3, 4, 5],
    'n_estimators': [150, 200, 250],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9, 1.0]
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, cv=5, scoring='r2', verbose=1)

# Fit the grid search to the data
grid_search.fit(X, y)

# Print the best parameters and the best score
print("Best parameters found:", grid_search.best_params_)
print("Best R-squared found:", grid_search.best_score_)


# The Training R² of 0.9557 and Test R² of 0.9445 are both very high, which indicates that the model explains a significant portion of the variance in both the training and test datasets. It's a sign of a strong model that is fitting well without major signs of overfitting since the test score is close to the training score.
# 
# Cross-Validation R² scores are also strong, with a mean of approximately 0.8525. While this is slightly lower than the R² on our test set, it's not uncommon for the cross-validation score to be a bit more conservative. This is because cross-validation averages the performance across multiple subsets of our data, providing a more generalized performance metric.
# 
# The Cross-Validation MSE has a mean of approximately 269.37, which is higher than our test MSE. This is expected because the test MSE is calculated on one fixed test set, while the cross-validation MSE is averaged over multiple folds and is generally a more robust estimate of the model's performance on unseen data.
# 
# The best hyperparameters from the grid search suggest a slightly lower learning rate and a higher max depth than our initial model. Interestingly, the best R-squared found from the grid search is very close to the mean CV R² score, suggesting that the parameters are well-tuned.
# 
# Overall, our model is performing consistently across different subsets of the data, and the hyperparameters seem well-chosen and the performance is strong.

# ### SVR(Support Vector Regressor) Model

# In[61]:


from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Define the independent variables (features) and the dependent variable (target)
features = X  # Ensure this is your dataset after preprocessing and PCA transformation
target = y  # Ensure this is your target variable, such as 'Appliances'

# Split the dataset into training and testing sets with a 80/20 split
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=85)

# Instantiate the SVR model with a Radial Basis Function (RBF) kernel
support_vector_regressor = SVR(kernel='rbf')

# Train the SVR model on the training data
support_vector_regressor.fit(X_train, Y_train)

# Make predictions on both the training and test sets
predictions_train = support_vector_regressor.predict(X_train)
predictions_test = support_vector_regressor.predict(X_test)

# Calculate the performance metrics for both the training and test sets
mse_train = mean_squared_error(Y_train, predictions_train)
mse_test = mean_squared_error(Y_test, predictions_test)

r2_train = r2_score(Y_train, predictions_train)
r2_test = r2_score(Y_test, predictions_test)

# Display the performance metrics
print("Support Vector Regressor Performance:")
print(f"Train MSE: {mse_train:.2f}")
print(f"Test MSE: {mse_test:.2f}")
print(f"Train R^2: {r2_train:.2f}")
print(f"Test R^2: {r2_test:.2f}")


# ### Cross Validation and hyper parameter tuning

# In[62]:


from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVR

# Assuming features (X) and target (y) are already defined and preprocessed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=85)

# Initialize SVR with default parameters to perform cross-validation
svr = SVR()

# Perform cross-validation
cv_scores = cross_val_score(svr, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-validation R2 scores: {cv_scores}")
print(f"Mean CV R2: {cv_scores.mean()}")

# Hyperparameter tuning using GridSearchCV
parameters = {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.01, 0.1, 1]}
grid_search = GridSearchCV(SVR(), parameters, cv=5, scoring='r2', verbose=1)
grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validated R2 score:", grid_search.best_score_)

# Train the final model with the best parameters found
best_svr = SVR(**grid_search.best_params_)
best_svr.fit(X_train, y_train)

# Evaluate the final model on the training set
train_mse = mean_squared_error(y_train, best_svr.predict(X_train))
train_r2 = r2_score(y_train, best_svr.predict(X_train))

# Evaluate the final model on the test set
test_mse = mean_squared_error(y_test, best_svr.predict(X_test))
test_r2 = r2_score(y_test, best_svr.predict(X_test))

# Print the perhttp://localhost:8890/notebooks/Desktop/ML_Project/Team8_Project.ipynb#Elastic-net-Regressionformance metrics
print(f"Train MSE: {train_mse:.2f}, Train R2: {train_r2:.2f}")
print(f"Test MSE: {test_mse:.2f}, Test R2: {test_r2:.2f}")


# ### Elastic net Regression

# In[63]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# Split the data into training and test sets with a 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=85)

# Generate polynomial and interaction features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Initialize ElasticNet regression model with default parameters
elastic_net_regressor = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=85)

# Fit ElasticNet model on the polynomially transformed training data
elastic_net_regressor.fit(X_train_poly, y_train)

# Predict on the transformed training and test sets
y_train_pred = elastic_net_regressor.predict(X_train_poly)
y_test_pred = elastic_net_regressor.predict(X_test_poly)

# Calculate and print the Mean Squared Error and R-squared values for both sets
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"ElasticNet Regressor Training MSE: {mse_train:.2f}")
print(f"ElasticNet Regressor Test MSE: {mse_test:.2f}")
print(f"ElasticNet Regressor Training R^2: {r2_train:.2f}")
print(f"ElasticNet Regressor Test R^2: {r2_test:.2f}")


# In[64]:


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Create a pipeline that first transforms the data using PolynomialFeatures and then fits an ElasticNet model
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('elastic_net', ElasticNet(random_state=85))
])

# Define a grid of hyperparameters to search over
param_grid = {
    'elastic_net__alpha': [ 0.1, 1, 10],
    'elastic_net__l1_ratio': np.linspace(0.1, 0.9, 9)
}

# Set up the grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', verbose=1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best combination of parameters
print('Best parameters found:', grid_search.best_params_)

# Print the best R^2 score found
print('Best R^2 score found:', grid_search.best_score_)

# Use the best estimator to make predictions
y_train_pred = grid_search.best_estimator_.predict(X_train)
y_test_pred = grid_search.best_estimator_.predict(X_test)

# Calculate and print the performance metrics for the best estimator
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training MSE: {train_mse:.2f}, Training R^2: {train_r2:.2f}")
print(f"Test MSE: {test_mse:.2f}, Test R^2: {test_r2:.2f}")


# ### Random Forest

# In[65]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Assuming X and y have already been defined and preprocessed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=85)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=85)

# Fit the regressor to the training data
rf_regressor.fit(X_train, y_train)

# Predict on the training and test data
y_pred_train_rf = rf_regressor.predict(X_train)
y_pred_test_rf = rf_regressor.predict(X_test)

# Calculate the performance metrics for training and test sets
mse_train_rf = mean_squared_error(y_train, y_pred_train_rf)
mse_test_rf = mean_squared_error(y_test, y_pred_test_rf)

r2_train_rf = r2_score(y_train, y_pred_train_rf)
r2_test_rf = r2_score(y_test, y_pred_test_rf)

# Print the performance metrics
print("Performance Metrics for Random Forest Regressor:")
print(f"Training MSE: {mse_train_rf:.2f}")
print(f"Test MSE: {mse_test_rf:.2f}")
print(f"Training R^2: {r2_train_rf:.4f}")
print(f"Test R^2: {r2_test_rf:.4f}")


# In[66]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=85)

# Perform cross-validation to evaluate the model's performance
cv_scores = cross_val_score(rf_regressor, X_train, y_train, cv=3, scoring='r2')  # Reduced to 3-fold CV
print(f"Cross-validation R2 scores: {cv_scores}")
print(f"Mean CV R2: {cv_scores.mean()}")

# Define a smaller grid of hyperparameters to search over
param_distributions = {
    'n_estimators': [50, 100],
    'max_depth': [None, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Set up randomized search with cross-validation
random_search = RandomizedSearchCV(
    rf_regressor, param_distributions, n_iter=10, cv=3, scoring='r2', verbose=1, random_state=85, n_jobs=-1
)
random_search.fit(X_train, y_train)

# Print the best combination of parameters and the corresponding R2 score
print("Best parameters found:", random_search.best_params_)
print("Best cross-validated R2 score:", random_search.best_score_)

# Train the final model with the best parameters found
best_rf = RandomForestRegressor(**random_search.best_params_, random_state=85)
best_rf.fit(X_train, y_train)

# Evaluate the final model on the test set
test_mse = mean_squared_error(y_test, best_rf.predict(X_test))
test_r2 = r2_score(y_test, best_rf.predict(X_test))

# Print the performance metrics for the test set
print(f"Test MSE: {test_mse:.2f}, Test R2: {test_r2:.2f}")


# In[ ]:




