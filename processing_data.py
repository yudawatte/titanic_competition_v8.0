"""
Read data from input csv files, clean data, alter features, and save
processed data.
"""
from settings import Settings
import helper
import pandas as pd
import numpy as np
from feature_handler import predict_missing_age_values, detect_outliers
from scipy import stats
from sklearn.decomposition import PCA

# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')

sett = Settings()

# Load input data
train = helper.read_data(sett.INPUT_DATA_PATH, 'train.csv')
test = helper.read_data(sett.INPUT_DATA_PATH, 'test.csv')

# Drop outliers in the train set based on 'Age', 'SibSp', 'Parch', 'Fare'
outlier_indices = detect_outliers(train, 2 , ["Age","SibSp","Parch","Fare"])
print("Outlier indices", outlier_indices)

print("Before deleting - train set:", train.shape)
train = train.drop(outlier_indices)
print("After deleting - train set:", train.shape)


# Joining two data sets
train_len = len(train)
dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

# Fill empty amd null values with Nan
dataset = dataset.fillna(np.nan)

# Feature: Fare
#Fill Fare missing values with the median value
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())

# Apply log to Fare to reduce skewness distribution
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

# Feature: Embarked
#Fill Embarked null values of dataset set with 'S' most frequent value
dataset["Embarked"] = dataset["Embarked"].fillna("S")

# Feature: Sex
# convert Sex into categorical value 0 for male and 1 for female
#dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":5})

# Feature: Age
# Filling missing value of Age
## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
"""
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) &
                               (dataset['Parch'] == dataset.iloc[i]["Parch"]) &
                               (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med
"""

# Predict missing age values
dataset = predict_missing_age_values(dataset)

# Categorized Age into 5 groups
dataset['AgeBin'] = pd.cut(dataset['Age'], bins=[0, 18, 35, 49, 60, 120],
                           labels=False)

# NewFeature AgeSexGroup
dataset['AgeSexGroup'] = dataset['Sex'] + dataset['AgeBin']

# Reset Female Sex value to 1 and male to 0
dataset['Sex'] = dataset['Sex'].map(lambda s: 1 if s == 5 else 0)

# Create new feature for age sex groups
dataset['M_Below_18'] = dataset['AgeSexGroup'].map(lambda s: 1 if s == 1 else 0)
dataset['M_19_35'] = dataset['AgeSexGroup'].map(lambda s: 1 if s == 2 else 0)
dataset['M_36-49'] = dataset['AgeSexGroup'].map(lambda s: 1 if s == 3 else 0)
dataset['M_50_60'] = dataset['AgeSexGroup'].map(lambda s: 1 if s == 4 else 0)
dataset['M_Above_60'] = dataset['AgeSexGroup'].map(lambda s: 1 if s == 5 else 0)

dataset['F_Below_18'] = dataset['AgeSexGroup'].map(lambda s: 1 if s == 6 else 0)
dataset['F_19_35'] = dataset['AgeSexGroup'].map(lambda s: 1 if s == 7 else 0)
dataset['F_36-49'] = dataset['AgeSexGroup'].map(lambda s: 1 if s == 8 else 0)
dataset['F_50_60'] = dataset['AgeSexGroup'].map(lambda s: 1 if s == 9 else 0)
dataset['F_Above_60'] = dataset['AgeSexGroup'].map(lambda s: 1 if s == 10 else 0)

# Drop variable: AgeSexGroup, AgeBin
dataset.drop(labels = ["AgeSexGroup"], axis = 1, inplace = True)
dataset.drop(labels = ["AgeBin"], axis = 1, inplace = True)

# New feature: Title
# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)

# Convert to categorical values Title
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)

# Drop variable: Name
dataset.drop(labels = ["Name"], axis = 1, inplace = True)

# New feature: Fsize
# Create a family size descriptor from SibSp and Parch
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1

# Create new feature of family size
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)

# convert to indicator values Title and Embarked
dataset = pd.get_dummies(dataset, columns = ["Title"])
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")

# Feature: Cabin
# Replace the Cabin number by the type of cabin 'X' if not
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])

# convert to indicator values Cabin
dataset = pd.get_dummies(dataset, columns = ["Cabin"], prefix="Cabin")

# Feature: Ticket
## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X.
Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit():
        Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0])  # Take prefix
    else:
        Ticket.append("X")

dataset["Ticket"] = Ticket

# convert to indicator values Ticket
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")

# Create categorical values for Pclass
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")

# Drop variable: PassangerId
dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)

# Normalize numeric features
"""dataset['Age'] = (dataset['Age'] - min(dataset['Age']))/(max(dataset['Age']) - min(dataset['Age']))
dataset['SibSp'] = (dataset['SibSp'] - min(dataset['SibSp']))/(max(dataset['SibSp']) - min(dataset['SibSp']))
dataset['Parch'] = (dataset['Parch'] - min(dataset['Parch']))/(max(dataset['Parch']) - min(dataset['Parch']))
dataset['Fare'] = (dataset['Fare'] - min(dataset['Fare']))/(max(dataset['Fare']) - min(dataset['Fare']))"""

dataset['Age'] = (dataset['Age'] - dataset['Age'].mean()) / dataset['Age'].std()
dataset['SibSp'] = (dataset['SibSp'] - dataset['SibSp'].mean()) / dataset['SibSp'].std()
dataset['Parch'] = (dataset['Parch'] - dataset['Parch'].mean()) / dataset['Parch'].std()
dataset['Fare'] = (dataset['Fare'] - dataset['Fare'].mean()) / dataset['Fare'].std()

## Separate train dataset and test dataset
train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels=["Survived"],axis = 1,inplace=True)

## Separate train features and label
train["Survived"] = train["Survived"].astype(int)
y_train = train["Survived"]
X_train = train.drop(labels = ["Survived"],axis = 1)
X_test = test

# Save processed data
helper.save_data(X_train, sett.PROCESSED_DATA_PATH, 'X_train.csv', header=True)
helper.save_data(y_train, sett.PROCESSED_DATA_PATH, 'y_train.csv', header=True)
helper.save_data(X_test, sett.PROCESSED_DATA_PATH, 'X_test.csv', header=True)

# Dimensionality reduction
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)

d_95 = np.argmax(cumsum >= 0.95) + 1
print("Number of variables when PCA with 95% :", d_95)

d_975 = np.argmax(cumsum >= 0.975) + 1
print("Number of variables when PCA with 97.5% :", d_975)

d_99 = np.argmax(cumsum >= 0.99) + 1
print("Number of variables when PCA with 99% :", d_99)

# PCA with 95%
pca = PCA(n_components=d_95)
X_train_95 = pca.fit_transform(X_train)
X_test_95 = pca.fit_transform(X_test)
helper.save_data(X_train_95, sett.PROCESSED_DATA_PATH, 'X_train_95.csv', header=True)
helper.save_data(X_test_95, sett.PROCESSED_DATA_PATH, 'X_test_95.csv', header=True)

# PCA with 97.5%
pca = PCA(n_components=d_975)
X_train_97_5 = pca.fit_transform(X_train)
X_test_97_5 = pca.fit_transform(X_test)
helper.save_data(X_train_97_5, sett.PROCESSED_DATA_PATH, 'X_train_97_5.csv', header=True)
helper.save_data(X_test_97_5, sett.PROCESSED_DATA_PATH, 'X_test_97_5.csv', header=True)

# PCA with 99%
pca = PCA(n_components=d_99)
X_train_99 = pca.fit_transform(X_train)
X_test_99 = pca.fit_transform(X_test)
helper.save_data(X_train_99, sett.PROCESSED_DATA_PATH, 'X_train_99.csv', header=True)
helper.save_data(X_test_99, sett.PROCESSED_DATA_PATH, 'X_test_99.csv', header=True)

print("\nX_train :", X_train.shape)
print("X_test :", X_test.shape)
print("\nX_train_95 :", X_train_95.shape)
print("X_test_95 :", X_test_95.shape)
print("\nX_train_97_5 :", X_train_97_5.shape)
print("X_test_97_5 :", X_test_97_5.shape)
print("\nX_train_99 :", X_train_99.shape)
print("X_test_99 :", X_test_99.shape)





