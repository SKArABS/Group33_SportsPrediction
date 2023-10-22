# %%
import sklearn
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split

# %% [markdown]
# **Importing Datasets**
# 
# **players_21 and players_22**

# %%
p21 = pd.read_csv('players_21.csv')
p22 = pd.read_csv('players_22.csv')

# %% [markdown]
# **Checking for NAN Values**

# %%
p21.info(verbose=True)

# %%
useless_columns = ['sofifa_id', 'player_url','short_name','long_name','player_positions','dob','club_name','league_name','club_loaned_from','club_joined','club_position','nationality_name','nation_position','nation_team_id','nation_jersey_number','body_type','real_face','player_tags','player_traits','player_face_url','club_logo_url','club_flag_url','nation_logo_url','nation_flag_url']

# %%
p21.drop(useless_columns, axis = 1, inplace=True)

# %%
p21.info(verbose=True)

# %% [markdown]
# **Seperating the object columns from the dataset**

# %%
p21_object_columns = p21.select_dtypes(include= ['object'])

# %% [markdown]
# **Seperating the numeric columns from the dataset**

# %%
p21_numeric_columns = p21.select_dtypes(include= ['int', 'int64', 'float'])

# %%
p21_position_values = p21_object_columns.drop(['preferred_foot','work_rate'], axis = 1)
p21_object_columns.drop(p21_position_values.columns, axis = 1, inplace=True)

# %%
p21_position_values

# %% [markdown]
# **Handling positions_values column**

# %%
def remove_modifiers(s):
    if '+' in s:
        return s.split('+')[0]
    if '-' in s:
        return s.split('-')[0]
    return s

# Apply the custom function to all columns using applymap
p21_position_values = p21_position_values.applymap(remove_modifiers)

# Print the modified DataFrame
p21_position_values

# %%
#Converting the position values into int type
p21_position_values = p21_position_values.astype('int64')

# %%
#Checking the number of null values in each column
p21_position_values.isnull().sum()

# %% [markdown]
# **Handling the numeric column**

# %%
#Checking the percentage of nan values in the numeric column
nan_percentage = (p21_numeric_columns.isna().sum() / len(p21_numeric_columns)) * 100
nan_percentage

# %%
p21_numeric_columns.drop(['goalkeeping_speed'], axis = 1, inplace=True) #Removing because it has a high percentage of nan values

# %%
#Filling nan values by imputation with Mean/Median
p21_numeric_columns.fillna(p21_numeric_columns.mean(), inplace=True)

# %%
p21_numeric_columns.isna().sum()

# %% [markdown]
# **Handling the object column**

# %%
#Checking for nan values in the object column
p21_object_columns.isna().sum()

# %%
#One-hot encoding for 'Preferred foot' and 'Work rate'
p21_object_columns_encoded = pd.get_dummies(p21_object_columns, p21_object_columns.columns, dtype= int)

p21_object_columns_encoded.head()

# %% [markdown]
# **Combining the refined Object and Numeric columns**

# %%
#The dataframe that we will be using to train our model
p21_refined = pd.concat([p21_numeric_columns, p21_object_columns_encoded,p21_position_values],axis = 1)

p21_refined.info()

# %%
#Scaling the new dataframe
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
p21_refined_scaled = scaler.fit_transform(p21_refined)
p21_refined_scaled_df = pd.DataFrame(p21_refined_scaled, columns =p21_refined.columns)
p21_refined_scaled_df

# %%
p21_correlation = p21_refined_scaled_df.corr(numeric_only=True, method = 'spearman')

# %%
pd.set_option('display.max_rows', 100)  # To display all correlations
p21_correlation['overall'].sort_values()

# %%
p21_corr_df = pd.DataFrame(p21_correlation['overall'].sort_values())
p21_corr_df = p21_corr_df.transpose()
p21_corr_df

# %%
threshold = 0.2
columns_to_drop = [col for col in corr_df.columns if (corr_df[col] < threshold).any()]
columns_to_drop
p21_refined_scaled_df.drop(columns_to_drop, axis = 1, inplace=True)
p21_refined_scaled_df

# %% [markdown]
# **Training the Model**

# %%
#Training with a VotingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

# %%
#Loading the data
X = p21_refined_scaled_df.drop(['overall'], axis = 1)
y = p21_refined_scaled_df['overall']

# %%
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# **Training with VotingRegressor using LinearRegression, SVR, KNeighborsRegressor**

# %%
# Create individual regression models
svr = SVR()
lr = LinearRegression()
knn_regressor = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors

# %%
# Create a VotingRegressor using the individual models
voting_regressor = VotingRegressor(estimators=[('SupportVector', svr),
                                              ('linear_regression', lr),
                                              ('knn', knn_regressor)])

# Train the VotingRegressor on the training data
voting_regressor.fit(X_train, y_train)

# %%
# Make predictions on the test set
y_pred = voting_regressor.predict(X_test)

# %%
from sklearn.model_selection import cross_val_score

# %%
#Performing cross-validation on the model
scores = cross_val_score(voting_regressor, X, y, cv=5, scoring='neg_mean_squared_error')

# Calculate the mean and standard deviation of the scores
mean_score = scores.mean()
std_score = scores.std()

print("Cross-Validation Scores: ", scores)
print("Mean Score: ", mean_score)
print("Standard Deviation: ", std_score)

# %% [markdown]
# **Training using a Random Forest Regressor**

# %%
from sklearn.ensemble import RandomForestRegressor
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor
RFR = RandomForestRegressor(n_estimators=500, random_state=42)  # You can adjust hyperparameters

# Train the model on the training data
RFR.fit(X_train, y_train)

# Make predictions on the test set
y_pred = RFR.predict(X_test)

# %%
#Performing cross-validation on the model
scores = cross_val_score(RFR, X, y, cv=5, scoring='neg_mean_squared_error')

# Calculate the mean and standard deviation of the scores
mean_score = scores.mean()
std_score = scores.std()

print("Cross-Validation Scores: ", scores)
print("Mean Score: ", mean_score)
print("Standard Deviation: ", std_score)

# %%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Create a Gradient Boosting Regressor
gbr = GradientBoostingRegressor(init=svr, n_estimators=500, learning_rate=0.1, max_depth=3)

# Train the model on the training data
gbr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gbr.predict(X_test)

# %%
#Performing cross-validation on the model
scores = cross_val_score(gbr, X, y, cv=5, scoring='neg_mean_squared_error')

# Calculate the mean and standard deviation of the scores
mean_score = scores.mean()
std_score = scores.std()

print("Cross-Validation Scores: ", scores)
print("Mean Score: ", mean_score)
print("Standard Deviation: ", std_score)

# %% [markdown]
# **Testing**

# %%
p22.drop(useless_columns, axis = 1, inplace=True)

# %%
p22.info(verbose=True)

# %%
p22_object_columns = p22.select_dtypes(include= ['object'])

# %%
p22_numeric_columns = p22.select_dtypes(include= ['int', 'int64', 'float'])

# %%
p22_position_values = p22_object_columns.drop(['preferred_foot','work_rate'], axis = 1)
p22_object_columns.drop(p22_position_values.columns, axis = 1, inplace=True)

# %%
p22_position_values

# %%
p22_position_values = p22_position_values.applymap(remove_modifiers)

# Print the modified DataFrame
p22_position_values

# %%
#Converting the position values into int type
p22_position_values = p22_position_values.astype('int64')

# %%
#Checking the number of null values in each column
p22_position_values.isnull().sum()

# %%
#Checking the percentage of nan values in the numeric column
nan_percentage = (p22_numeric_columns.isna().sum() / len(p22_numeric_columns)) * 100
nan_percentage

# %%
p22_numeric_columns.drop(['goalkeeping_speed'], axis = 1, inplace=True) #Removing because it has a high percentage of nan values

# %%
#Filling nan values by imputation with Mean/Median
p22_numeric_columns.fillna(p22_numeric_columns.mean(), inplace=True)

# %%
#Checking for nan values in the object column
p22_object_columns.isna().sum()

# %%
#One-hot encoding for 'Preferred foot' and 'Work rate'
p22_object_columns_encoded = pd.get_dummies(p22_object_columns, p22_object_columns.columns, dtype= int)

p22_object_columns_encoded.head()

# %%
#The dataframe that we will be using to train our model
p22_refined = pd.concat([p22_numeric_columns, p22_object_columns_encoded,p22_position_values],axis = 1)

p22_refined.info()

# %%
#Scaling the new dataframe
p22_refined_scaled = scaler.fit_transform(p22_refined)
p22_refined_scaled_df = pd.DataFrame(p22_refined_scaled, columns =p22_refined.columns)
p22_refined_scaled_df

# %%
p22_refined_scaled_df = p22_refined_scaled_df[p21_refined_scaled_df.columns]

# %%
p22_refined_scaled_df

# %%
#Loading the data
X = p22_refined_scaled_df.drop(['overall'], axis = 1)
y = p22_refined_scaled_df['overall']

# %%
from sklearn.ensemble import RandomForestRegressor
# Split the data into a training set and a test set
X_test = p22_refined_scaled_df.drop(['overall'], axis = 1)

# Make predictions on the test set
y_pred = RFR.predict(X_test)


# %%
#Performing cross-validation on the model
scores = cross_val_score(RFR, X, y, cv=5, scoring='neg_mean_squared_error')

# Calculate the mean and standard deviation of the scores
mean_score = scores.mean()
std_score = scores.std()

print("Cross-Validation Scores: ", scores)
print("Mean Score: ", mean_score)
print("Standard Deviation: ", std_score)

# %% [markdown]
# **Saving the model**

# %%
import pickle


# %%
with open('FIFA_Rating_Generator.pkl', 'wb') as file:
    pickle.dump(RFR, file)



