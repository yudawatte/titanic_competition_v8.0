"""
Fit Ada Boost Classifier model on the preprocessed data
"""
from settings import Settings
from model_processor import fit_ml_algo, show_performance
import helper
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')

sett = Settings()

# Read data
X_train = helper.read_data(sett.PROCESSED_DATA_PATH, 'X_train.csv')
X_test = helper.read_data(sett.PROCESSED_DATA_PATH, 'X_test.csv')
y_train = helper.read_data(sett.PROCESSED_DATA_PATH, 'y_train.csv')
X_test_org = helper.read_data(sett.INPUT_DATA_PATH, 'test.csv')

# Select best 15 features
# 'Sex', 'Title_2', 'Pc_3', 'Fare', 'Fsize', 'F_Below_18', 'M_Below_18', 'Title_1', 'Pc_2', 'LargeF', 'Pc_1', 'Title_0', 'MedF', 'SibSp', 'Em_S'
X_train_reduced = X_train[['Sex', 'Title_2', 'Pc_3', 'Fare', 'Fsize', 'F_Below_18', 'M_Below_18', 'Title_1', 'Pc_2', 'LargeF', 'Pc_1', 'Title_0', 'MedF', 'SibSp', 'Em_S']]
X_test_reduced = X_test[['Sex', 'Title_2', 'Pc_3', 'Fare', 'Fsize', 'F_Below_18', 'M_Below_18', 'Title_1', 'Pc_2', 'LargeF', 'Pc_1', 'Title_0', 'MedF', 'SibSp', 'Em_S']]

# Fine-tuning model
# First do a high level tune-up with RandomizedSearchCV
"""param_grid = {
    'n_estimators': [100, 300, 500, 600, 700],
    'learning_rate':[0.001, 0.01, 0.1, 1],
    'algorithm': ['SAMME', 'SAMME.R'],
}
clf_ada = RandomizedSearchCV(AdaBoostClassifier(DecisionTreeClassifier(random_state=42)),
                             random_state=42,
                             param_distributions=param_grid,
                             cv=10,
                             n_jobs=-1)

best_clf_ada = clf_ada.fit(X_train_reduced, y_train.values.ravel())
show_performance(best_clf_ada, "Ada Boost Classifier")"""

"""
Ada Boost Classifier - RandomizedSearchCV results
Best Score:	 0.8183605720122575
Best Parameters:	 {'n_estimators': 300, 'learning_rate': 0.01, 'algorithm': 'SAMME'}
"""

# Fine tuning with GridSearchCV
param_grid = {'n_estimators': np.arange(295, 305, 1),
              'learning_rate': np.arange(0.008, 0.012, 0.01),
              'algorithm': ['SAMME', 'SAMME.R']}
clf_ada = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier(random_state=2),random_state=2,learning_rate=0.1),
                       param_grid=param_grid, cv=10, verbose=True, n_jobs=-1)
best_clf_ada = clf_ada.fit(X_train_reduced, y_train.values.ravel())
show_performance(best_clf_ada, "Ada Boost Classifier")

"""
Ada Boost Classifier - GridSearchCV results
Best Score:	 0.8206460674157304
Best Parameters:	 {'algorithm': 'SAMME', 'learning_rate': 0.008, 'n_estimators': 296}
"""

# Predict on test set
predict = best_clf_ada.predict(X_test_reduced)

# Saving the model
print("\nSavning Model")
helper.save_model(best_clf_ada, "ada_model.sav")

# Save submit results
print("\nSaving submit results")
X_test_org = helper.read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
helper.save_data(base_submission, sett.RESULT_DATA_PATH, "ada_results.csv",
                 index=False, header=True)
