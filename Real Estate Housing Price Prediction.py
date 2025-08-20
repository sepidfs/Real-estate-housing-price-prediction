import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score


# Replace the path below with your actual file path if needed
file_path = r"C:\Users\14163\Desktop\university cu boulder\GorgeBrown\Mashine Learning 1\Assignment 6\Real estate.csv"

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Display the first 5 samples
df.head()


# ### Show more information about the dataset


# Show general info: column names, non-null counts, and data types
df.info()

# Show summary statistics for all numeric columns
df.describe()


 ### Find how many samples are there and how many columns are there in the dataset

num_rows, num_columns = df.shape

print(f"Number of samples (rows): {num_rows}")
print(f"Number of features (columns): {num_columns}")

# Show the list of all columns in the dataset
print("Features in the dataset:")
print(df.columns.tolist())


 ### Check if any features have missing data

print("Missing values per column:")
print(df.isnull().sum())

# Ensure "No" column is included in the features
X = df.drop(columns=["Y house price of unit area"])

# Confirm the shape is 414 rows and 7 columns
print(f"Shape of X: {X.shape}")

# Display the top rows
X.head()

### Group feature(s) as independent features in y


# Assign the target variable
y = df["Y house price of unit area"]

# Show structured view: first and last 5 values + summary
from IPython.display import display, Markdown

display(Markdown("###Target Variable: `Y house price of unit area`"))
display(Markdown(f"- Total samples: **{len(y)}**"))
display(Markdown("#### First 5 entries:"))
display(y.head())

display(Markdown("#### Last 5 entries:"))
display(y.tail())

display(Markdown("#### Data Type:"))
print(f"{y.name}, dtype: {y.dtype}")

 ### Split the dataset into train and test data

from sklearn.model_selection import train_test_split

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print the shapes to confirm
print(f"Training set:   X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Testing set:    X_test  = {X_test.shape}, y_test  = {y_test.shape}")


 ### Choose the model (Linear Regression)

from sklearn.linear_model import LinearRegression

# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Confirm training completion
print("Linear Regression model has been trained successfully.")

#  Create an Estimator Object
from sklearn.linear_model import LinearRegression

# Create the estimator (Linear Regression model)
estimator = LinearRegression()

# Fit the model to training data
estimator.fit(X_train, y_train)

print("Estimator object created and model fitted.")

 ### Train the model

estimator.fit(X_train, y_train)

print(" Model trained successfully on the training set.")


 ### Apply the model and show predictions as a NumPy array
y_pred = estimator.predict(X_test)

# Show predictions in array format
print("array :", np.array2string(np.array(y_pred), separator=', ', threshold=np.inf))


# ### Display the coefficients

print("array(", end="")
print(np.array2string(estimator.coef_, separator=', ', precision=8, suppress_small=False), end=")\n")


### Find how well the trained model did with testing data

# Predict again to be safe
y_pred = estimator.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Display results
print(f"R² Score:  {r2:.4f}")
print(f"RMSE:      {rmse:.4f}")

 ### Plot House Age Vs Price

# Set the plot size and style
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Create the regression plot
sns.regplot(x="X2 house age", y="Y house price of unit area", data=df)

# Add labels and title
plt.xlabel("House Age (years)")
plt.ylabel("House Price per Unit Area")
plt.title("House Age vs. House Price")

# Show the plot
plt.show()

### Plot Distance to MRT station Vs Price

# Set the plot size and style
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Create the regression plot
sns.regplot(x="X3 distance to the nearest MRT station", y="Y house price of unit area", data=df)

# Add labels and title
plt.xlabel("Distance to Nearest MRT Station (meters)")
plt.ylabel("House Price per Unit Area")
plt.title("MRT Distance vs. House Price")

# Show the plot
plt.show()

 ### Plot Number of Convienience Stores Vs Price

# Set plot size and style
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Create regression plot
sns.regplot(
    x="X4 number of convenience stores",
    y="Y house price of unit area",
    data=df
)

# Add labels and title
plt.xlabel("Number of Nearby Convenience Stores")
plt.ylabel("House Price per Unit Area")
plt.title("Convenience Stores vs. House Price")

# Show the plot
plt.show()

