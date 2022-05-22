"""
Fit Extra Tree Classifier model on the scaled, encoded data
"""
from settings import Settings
from model_processor import fit_ml_algo, show_performance
import helper
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import ExtraTreesClassifier
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

# Fine tuning the model
"""param_grid = {
    'n_estimators': [450, 500, 550, 600,650],
    'criterion':['gini', 'entropy'],
    'min_samples_split':[2, 3],
    'min_samples_leaf':[2, 3],
    'max_depth': [20, 25, 30, 35],
    'bootstrap': [True, False]}

rnd_etc = RandomizedSearchCV(ExtraTreesClassifier(random_state=42),
                            param_distributions=param_grid,
                            cv=10,
                            n_jobs=-1)

best_clf_etc = rnd_etc.fit(X_train_reduced, y_train.values.ravel())
show_performance(best_clf_etc, "Extra Tree Classifier")"""

"""
Extra Tree Classifier - RandomizedSearchCV results
Extra Tree Classifier
Best Score:	 0.8252298263534218
Best Parameters:	 {'n_estimators': 650, 
                    'min_samples_split': 2, 
                    'min_samples_leaf': 2,
                    'max_depth': 35, 
                    'criterion': 'entropy', 
                    'bootstrap': False}
"""
# Fine tuning with GridSearchCV
param_grid = {'n_estimators': np.arange(645, 655, 1),
              'criterion': ['entropy'],
              'min_samples_split': [2],
              'min_samples_leaf': [2],
              'max_depth': np.arange(33, 37, 1),
              'bootstrap': [False]}

grid_etc = GridSearchCV(ExtraTreesClassifier(random_state=2),
                        param_grid=param_grid, cv=10, verbose=True, n_jobs=-1)
best_clf_etc = grid_etc.fit(X_train_reduced, y_train.values.ravel())
show_performance(best_clf_etc, "Extra Tree Classifier")

"""
Extra Tree Classifier - GridSearchCV results
Best Score:	 0.8308605720122575
Best Parameters:	 {'bootstrap': False, 
                    'criterion': 'entropy', 
                    'max_depth': 33, 
                    'min_samples_leaf': 2, 
                    'min_samples_split': 2, 
                    'n_estimators': 650}
"""

# Predict on test set
predict = best_clf_etc.predict(X_test_reduced)

# Saving the model
print("\nSavning Model")
helper.save_model(best_clf_etc, "etc_model.sav")

# Save submit results
print("\nSaving submit results")
X_test_org = helper.read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
helper.save_data(base_submission, sett.RESULT_DATA_PATH, "etc_results.csv",
                 index=False, header=True)