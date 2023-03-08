import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load the train data
train_data = pd.read_csv("train.csv")

# Fill in missing values in the Age column
imputer = SimpleImputer()
train_data["Age"] = imputer.fit_transform(train_data[["Age"]])

# Fill in missing values in the Fare column with the median value
train_data["Fare"].fillna(train_data["Fare"].median(), inplace=True)

# Convert "Sex" column into binary values using one-hot encoding
train_data = pd.get_dummies(train_data, columns=["Sex"])

# Define the features to use in the model
features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_female", "Sex_male"]

# Split the train data into features and target
train_features = train_data[features]
train_target = train_data["Survived"]

# Create a Random Forest Classifier model
clf = RandomForestClassifier(n_estimators=100)

# Train the model
clf.fit(train_features, train_target)

# Load the test data
test_data = pd.read_csv("test.csv")

# Fill in missing values in the Age column
test_data["Age"] = imputer.transform(test_data[["Age"]])

# Fill in missing values in the Fare column with the median value
test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)

# Convert "Sex" column into binary values using one-hot encoding
test_data = pd.get_dummies(test_data, columns=["Sex"])

# Select the features for the test data
test_features = test_data[features]

# Make predictions on the test data
predictions = clf.predict(test_features)

# Save the predictions to a CSV file
output = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": predictions})
output.to_csv("submission.csv", index=False)
