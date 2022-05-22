"""
Fit Random Forest Classifier model on the scaled, encoded data
"""
from settings import Settings
from model_processor import fit_ml_algo, show_performance
import helper
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from feature_handler import detect_feature_importance

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
    'n_estimators': [100, 200, 300, 400, 500, 600],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False],
    'max_depth': [15, 20, 25],
    #'max_features': ['auto', 'sqrt', 10, 20, 30],
    'min_samples_leaf': [2, 3],
    'min_samples_split': [2, 3],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}
rnd_rf = RandomizedSearchCV(RandomForestClassifier(random_state=2),
                            param_distributions=param_grid,
                            cv=10, verbose=True, n_jobs=-1)

best_clf_rf = rnd_rf.fit(X_train_reduced, y_train.values.ravel())
show_performance(best_clf_rf, "Random Forest Classifier")"""

"""
Random Forest Classifier - RandomizedSearchCV results
With best 10 features
Best Score:	 0.8308861082737486
Best Parameters:	 {'n_estimators': 200, 
                    'min_samples_split': 3, 
                    'min_samples_leaf': 2, 
                    'max_depth': 25, 
                    'criterion': 'gini', 
                    'class_weight': None, 
                    'bootstrap': True}
"""

# Fine tuning with GridSearchCV
param_grid = {'n_estimators': np.arange(190, 210, 1),
              'criterion':['gini'],
              'bootstrap': [True],
              'max_depth': np.arange(23, 27, 1),
              #'max_features': [10],
              'min_samples_leaf': [3],
              'min_samples_split': [2]}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid,
                       cv=10, verbose=True, n_jobs=-1)
best_clf_rf = grid_rf.fit(X_train_reduced, y_train.values.ravel())
show_performance(best_clf_rf, "Random Forest Classifier")

"""
Random Forest Classifier - GridSearchCV results
Best Score:	 0.8286133810010217
Best Parameters:	 {'bootstrap': True, 
                    'criterion': 'gini', 
                    'max_depth': 23, 
                    'min_samples_leaf': 3, 
                    'min_samples_split': 2, 
                    'n_estimators': 192}
"""

# Predict on test set
predict = best_clf_rf.predict(X_test_reduced)

# Saving the model
print("\nSavning Model")
helper.save_model(best_clf_rf, "rf_model.sav")

# Save submit results
print("\nSaving submit results")
X_test_org = helper.read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
helper.save_data(base_submission, sett.RESULT_DATA_PATH, "rf_results.csv",
                 index=False, header=True)