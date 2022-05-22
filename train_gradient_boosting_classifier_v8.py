"""
Fit Gradient Boosting Classifier model on the scaled, encoded data
"""
from settings import Settings
from model_processor import fit_ml_algo, show_performance
import helper
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import GradientBoostingClassifier
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
    'loss': ['deviance', 'exponential'],
    'learning_rate':[0.001, 0.01, 0.1, 1],
    'n_estimators': [450, 500, 550, 600,650],
    'criterion':['friedman_mse', 'squared_error'],
    'min_samples_split':[2, 3],
    'min_samples_leaf':[2, 3],
    'max_depth': [20, 25, 30, 35]
}
rnd_gbc = RandomizedSearchCV(GradientBoostingClassifier(random_state=42),
                             param_distributions=param_grid,
                             cv=10,
                             verbose=False,
                             n_jobs=-1)
best_clf_gbc = rnd_gbc.fit(X_train_reduced, y_train.values.ravel())
show_performance(best_clf_gbc, "Gradient Boosting Classifier")"""

"""
Gradient Boosting Classifier - RandomizedSearchCV results
Best Score:	 0.8229443309499489
Best Parameters:	 {'n_estimators': 550, 
                    'min_samples_split': 3, 
                    'min_samples_leaf': 3, 
                    'max_depth': 35, 
                    'loss': 'exponential', 
                    'learning_rate': 0.001, 
                    'criterion': 'friedman_mse'}
"""
# Fine tuning with GridSearchCV
param_grid = {'loss': ['exponential'],
              'learning_rate': np.arange(0.0001, 0.0012, 0.0001),
              'n_estimators': np.arange(545,555, 1),
              'criterion':['friedman_mse'],
              'min_samples_split':[3],
              'min_samples_leaf':[3],
              'max_depth': np.arange(33, 38, 1)}

grid_gbc = GridSearchCV(GradientBoostingClassifier(random_state=42),
                        param_grid=param_grid, cv=10, verbose=True, n_jobs=-1)
best_clf_gbc = grid_gbc.fit(X_train_reduced, y_train.values.ravel())
show_performance(best_clf_gbc, "Gradient Boosting Classifier")

"""
Gradient Boosting Classifier - GridSearchCV results
Best Score:	 0.8274897854954034
Best Parameters:	 {'criterion': 'friedman_mse', 
                    'learning_rate': 0.0011, 
                    'loss': 'exponential', 
                    'max_depth': 33, 
                    'min_samples_leaf': 3, 
                    'min_samples_split': 3, 
                    'n_estimators': 551}
"""

# Predict on test set
predict = best_clf_gbc.predict(X_test_reduced)

# Saving the model
print("\nSavning Model")
helper.save_model(best_clf_gbc, "gbc_model.sav")

# Save submit results
print("\nSaving submit results")
X_test_org = helper.read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
print(X_test_org.shape)
print(predict.shape)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
helper.save_data(base_submission, sett.RESULT_DATA_PATH, "gbc_results.csv",
                 index=False, header=True)