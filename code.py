import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

# Step 1: Load and combine the earthquake datasets with a 'region' column
auckland_df = pd.read_csv('./project/earthquakes-auckland.csv')
auckland_df['region'] = 'Auckland'

canterbury_df = pd.read_csv('./project/earthquakes-cantebury.csv')
canterbury_df['region'] = 'Canterbury'

fiordland_df = pd.read_csv('./project/earthquakes-fiorland.csv')
fiordland_df['region'] = 'Fiordland'

gisborne_df = pd.read_csv('./project/earthquakes-gisborne.csv')
gisborne_df['region'] = 'Gisborne'

hawkes_bay_df = pd.read_csv("./project/earthquakes-Hawke's Bay.csv")
hawkes_bay_df['region'] = "Hawke's Bay"

nelson_df = pd.read_csv('./project/earthquakes-nelson.csv')
nelson_df['region'] = 'Nelson'

otago_southland_df = pd.read_csv("./project/earthquakes-Otago & Southland.csv")
otago_southland_df['region'] = 'Otago & Southland'

taranaki_df = pd.read_csv('./project/earthquakes-Taranaki.csv')
taranaki_df['region'] = 'Taranaki'

tongariro_bay_of_plenty_df = pd.read_csv('./project/earthquakes-Tongariro & Bay of Plenty.csv')
tongariro_bay_of_plenty_df['region'] = 'Tongariro_Bay_of_Plenty'

wellington_df = pd.read_csv('./project/earthquakes-wellington.csv')
wellington_df['region'] = 'Wellington'

combined_df = pd.concat([auckland_df, canterbury_df, fiordland_df, gisborne_df, hawkes_bay_df, 
                         nelson_df, otago_southland_df, taranaki_df, tongariro_bay_of_plenty_df, 
                         wellington_df])

# Step 2: Data preparation
cleaned_df = combined_df.drop(columns=['publicid', 'eventtype', 'modificationtime', 'magnitudetype', 
                                       'depthtype', 'evaluationstatus', 'evaluationmode', 'earthmodel'])
cleaned_df = cleaned_df.dropna()
cleaned_df['origintime'] = pd.to_datetime(cleaned_df['origintime'])
cleaned_df['year'] = cleaned_df['origintime'].dt.year
cleaned_df['month'] = cleaned_df['origintime'].dt.month
cleaned_df['day'] = cleaned_df['origintime'].dt.day
cleaned_df['hour'] = cleaned_df['origintime'].dt.hour
cleaned_df = cleaned_df.drop(columns=['origintime'])

# Ensure the 'region' column is excluded from the features
X_encoded = pd.get_dummies(cleaned_df, columns=['evaluationmethod'])

# Drop the 'magnitude' (target) and 'region' columns from X
X = X_encoded.drop(columns=['magnitude', 'region'])

# The target remains the same
y = cleaned_df['magnitude']

# Step 3: Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the models

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, y_rf_pred)
rf_r2 = r2_score(y_test, y_rf_pred)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_lr_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, y_lr_pred)
lr_r2 = r2_score(y_test, y_lr_pred)

# Gradient Boosting Model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_gb_pred = gb_model.predict(X_test)
gb_mse = mean_squared_error(y_test, y_gb_pred)
gb_r2 = r2_score(y_test, y_gb_pred)

# Step 5: Visualization (Comparison of MSE and R-squared for all models)
comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'Linear Regression', 'Gradient Boosting'],
    'MSE': [rf_mse, lr_mse, gb_mse],
    'R-squared': [rf_r2, lr_r2, gb_r2]
})

plt.figure(figsize=(10, 6))
plt.bar(comparison_df['Model'], comparison_df['MSE'], color=['blue', 'green', 'orange'])
plt.title('Model Comparison - Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.savefig('model_comparison_mse.png')

plt.figure(figsize=(10, 6))
plt.bar(comparison_df['Model'], comparison_df['R-squared'], color=['blue', 'green', 'orange'])
plt.title('Model Comparison - R-squared')
plt.ylabel('R-squared')
plt.savefig('model_comparison_r2.png')

# Step 6: Feature importance visualization for Random Forest
importances = rf_model.feature_importances_
features = X_train.columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), features[indices], rotation=90)
plt.tight_layout()
plt.savefig('feature_importances_rf.png')

# Step 7: New data prediction (generating random earthquake-like data)
new_data = pd.DataFrame({
    'longitude': np.random.uniform(cleaned_df['longitude'].min(), cleaned_df['longitude'].max(), 20),
    'latitude': np.random.uniform(cleaned_df['latitude'].min(), cleaned_df['latitude'].max(), 20),
    'depth': np.random.uniform(cleaned_df['depth'].min(), cleaned_df['depth'].max(), 20),
    'usedphasecount': np.random.randint(cleaned_df['usedphasecount'].min(), cleaned_df['usedphasecount'].max(), 20),
    'usedstationcount': np.random.randint(cleaned_df['usedstationcount'].min(), cleaned_df['usedstationcount'].max(), 20),
    'magnitudestationcount': np.random.randint(cleaned_df['magnitudestationcount'].min(), cleaned_df['magnitudestationcount'].max(), 20),
    'minimumdistance': np.random.uniform(cleaned_df['minimumdistance'].min(), cleaned_df['minimumdistance'].max(), 20),
    'azimuthalgap': np.random.uniform(cleaned_df['azimuthalgap'].min(), cleaned_df['azimuthalgap'].max(), 20),
    'originerror': np.random.uniform(cleaned_df['originerror'].min(), cleaned_df['originerror'].max(), 20),
    'magnitudeuncertainty': np.random.uniform(cleaned_df['magnitudeuncertainty'].min(), cleaned_df['magnitudeuncertainty'].max(), 20),
    'year': 2024,
    'month': np.random.randint(1, 12, 20),
    'day': np.random.randint(1, 31, 20),
    'hour': np.random.randint(0, 23, 20),
})

# Add missing evaluation method columns for new data
new_data['evaluationmethod'] = 'LOCSAT'
new_data_encoded = pd.get_dummies(new_data, columns=['evaluationmethod'])
new_data_encoded['evaluationmethod_FixedHypocenter'] = 0
new_data_encoded['evaluationmethod_NonLinLoc'] = 0

# Ensure that the feature names in the new data match the training data
missing_columns = set(X_train.columns) - set(new_data_encoded.columns)

# Add any missing columns with zeros
for col in missing_columns:
    new_data_encoded[col] = 0

# Ensure that the columns in new_data_encoded are in the same order as X_train
new_data_encoded = new_data_encoded[X_train.columns]

# Match data types (this step helps if there is a data type mismatch issue)
new_data_encoded = new_data_encoded.astype(X_train.dtypes.to_dict())

# Now predict using the trained models
rf_predictions = rf_model.predict(new_data_encoded)

# Predict using the trained models
rf_predictions = rf_model.predict(new_data_encoded)
lr_predictions = lr_model.predict(new_data_encoded)
gb_predictions = gb_model.predict(new_data_encoded)

# Step 8: Plot the predicted magnitudes for each model
plt.figure(figsize=(10, 6))
plt.plot(rf_predictions, label="Random Forest Predictions", color='blue')
plt.plot(lr_predictions, label="Linear Regression Predictions", color='green')
plt.plot(gb_predictions, label="Gradient Boosting Predictions", color='orange')
plt.title('Predicted Earthquake Magnitudes - New Data')
plt.xlabel('New Data Samples')
plt.ylabel('Predicted Magnitude')
plt.legend()
plt.grid(True)
plt.savefig('predicted_magnitudes_new_data.png')

# Step 9: Model Transparency Using LIME
explainer_lime = LimeTabularExplainer(X_train.values,feature_names=X_train.columns, class_names=['Magnitude'], mode='regression')

# Explain a specific prediction
exp = explainer_lime.explain_instance(X_test.iloc[0].values, rf_model.predict, num_features=10)
exp.save_to_file('lime_explanation.html')

# Step 10: Residual Plot for Random Forest Model (without lowess)
plt.figure(figsize=(8, 6))
sns.residplot(x=y_test, y=y_rf_pred, color="g")
plt.title('Residual Plot - Random Forest Model')
plt.xlabel('Actual Magnitude')
plt.ylabel('Residuals')
plt.grid(True)
plt.savefig('residual_plot_rf.png')

# Step 11: Comparison of Actual vs Predicted Magnitudes for All Models
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual Magnitude", color='black', linestyle='--')
plt.plot(y_rf_pred, label="Random Forest Predicted", color='blue')
plt.plot(y_lr_pred, label="Linear Regression Predicted", color='green')
plt.plot(y_gb_pred, label="Gradient Boosting Predicted", color='orange')
plt.title('Comparison of Actual vs Predicted Magnitudes')
plt.xlabel('Test Samples')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.savefig('comparison_actual_vs_predicted.png')

# Step 12: Distribution of Actual vs Predicted Magnitudes
plt.figure(figsize=(10, 6))
sns.histplot(y_test, color='blue', label='Actual Magnitudes', kde=True)
sns.histplot(y_rf_pred, color='orange', label='Predicted Magnitudes (RF)', kde=True)
plt.title('Distribution of Actual vs Predicted Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('distribution_actual_vs_predicted_magnitudes.png')

# Step 13: Prediction Error Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_rf_pred, alpha=0.5, color="blue", label="Random Forest Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2, label='Perfect Prediction')
plt.title('Prediction Error Plot - Random Forest')
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.legend()
plt.grid(True)
plt.savefig('prediction_error_plot_rf.png')

# Step 14: Investigate Bias in Code

# 14.1: Investigating magnitude bias
plt.figure(figsize=(10, 6))
sns.histplot(cleaned_df['magnitude'], bins=30, kde=True)
plt.title('Distribution of Earthquake Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.savefig('magnitude_distribution.png')

# Splitting both the features and the region column
X_train, X_test, y_train, y_test, region_train, region_test = train_test_split(
    X, y, cleaned_df['region'], test_size=0.2, random_state=42
)

# Step 14.2: Performance by Region
print("Analyzing Performance by Region...")
regions = cleaned_df['region'].unique()
for region in regions:
    X_region = X_test[region_test == region]
    y_region = y_test[region_test == region]
    
    # Skip region if there are no data points in the test set
    if X_region.shape[0] == 0:
        print(f"No data for region: {region}")
        continue

    y_region_pred = rf_model.predict(X_region)
    mse_region = mean_squared_error(y_region, y_region_pred)
    print(f'Region: {region}, MSE: {mse_region}')

print("Performance by Region Analysis Complete")

# Step 14.3: Performance by Magnitude Range
bins = pd.cut(y_test, bins=[0, 2, 4, 6, 8, 10], labels=['0-2', '2-4', '4-6', '6-8', '8-10'], include_lowest=True)

# Iterate through the unique bins, ignoring NaN
for b in bins.unique():
    if pd.isna(b):  # Skip NaN bins
        print(f"No valid samples in bin: {b}")
        continue
    
    X_bin = X_test[bins == b]
    y_bin = y_test[bins == b]
    
    # Check if the bin is empty before proceeding
    if X_bin.shape[0] == 0:
        print(f"No samples in bin: {b}")
        continue
    
    y_bin_pred = rf_model.predict(X_bin)
    mse_bin = mean_squared_error(y_bin, y_bin_pred)
    print(f'Magnitude Range: {b}, MSE: {mse_bin}')

