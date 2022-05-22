"""
Fit trained models with a voting classifier
"""
from sklearn.ensemble import VotingClassifier
from model_processor import fit_ml_algo, show_performance
from helper import read_data, save_data, save_model, load_model, show_results
from settings import Settings
from sklearn.model_selection import GridSearchCV
import time
import pandas as pd
import helper

# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')

sett = Settings()

# Read data
X_train = helper.read_data(sett.PROCESSED_DATA_PATH, 'X_train.csv')
X_test = helper.read_data(sett.PROCESSED_DATA_PATH, 'X_test.csv')
y_train = helper.read_data(sett.PROCESSED_DATA_PATH, 'y_train.csv')

# Select best 15 features
# 'Sex', 'Title_2', 'Pc_3', 'Fare', 'Fsize', 'F_Below_18', 'M_Below_18', 'Title_1', 'Pc_2', 'LargeF', 'Pc_1', 'Title_0', 'MedF', 'SibSp', 'Em_S'
X_train_reduced = X_train[['Sex', 'Title_2', 'Pc_3', 'Fare', 'Fsize', 'F_Below_18', 'M_Below_18', 'Title_1', 'Pc_2', 'LargeF', 'Pc_1', 'Title_0', 'MedF', 'SibSp', 'Em_S']]
X_test_reduced = X_test[['Sex', 'Title_2', 'Pc_3', 'Fare', 'Fsize', 'F_Below_18', 'M_Below_18', 'Title_1', 'Pc_2', 'LargeF', 'Pc_1', 'Title_0', 'MedF', 'SibSp', 'Em_S']]


# Load saved best models
print("\nLoading saved models")

rf = load_model("rf_model.sav")
ada = load_model("ada_model.sav")
etc = load_model("etc_model.sav")
gbc = load_model("gbc_model.sav")

# Voting classifier
# Soft voting
voting_clf_soft = VotingClassifier(
    estimators = [('rf', rf),
                  ('ada', ada),
                  ('etc', etc),
                  ('gbc', gbc)],
    verbose=True,
    voting='soft')

"""start_time = time.time()
train_pred_log, acc_log, acc_cv_log = fit_ml_algo(voting_clf,
                                                 X_train,
                                                 y_train.values.ravel(),
                                                 10)
log_time = (time.time() - start_time)
print("Initial model accuracy")
print("\tAccuracy: \t%s" % acc_log)
print("\tAccuracy CV 10-Fold: %s" % acc_cv_log)
print("\tRunning Time: \t%s" % log_time)
print("----------------------------------------\n")"""

voting_clf_soft.fit(X_train_reduced, y_train.values.ravel())
# Predict on test set
predict = voting_clf_soft.predict(X_test_reduced)

# Saving the model
print("\nSavning Model")
helper.save_model(voting_clf_soft, "vc_soft_model.sav")

# Save submit results
print("\nSaving submit results")
X_test_org = helper.read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
helper.save_data(base_submission, sett.RESULT_DATA_PATH, "vc_soft_results.csv", index=False, header=True)

# Hard voting
# Voting classifier
voting_clf_hard = VotingClassifier(
    estimators = [('rf', rf),
                  ('ada', ada),
                  ('etc', etc),
                  ('gbc', gbc)],
    verbose=True,
    voting='hard')

voting_clf_hard.fit(X_train_reduced, y_train.values.ravel())
# Predict on test set
predict = voting_clf_hard.predict(X_test_reduced)

# Saving the model
print("\nSavning Model")
helper.save_model(voting_clf_hard, "vc_hard_model.sav")

# Save submit results
print("\nSaving submit results")
X_test_org = helper.read_data(sett.INPUT_DATA_PATH, sett.TEST_SET_FILENAME)
basic_submission = {'PassengerId': X_test_org.PassengerId, 'Survived': predict}
base_submission = pd.DataFrame(data=basic_submission)
helper.save_data(base_submission, sett.RESULT_DATA_PATH, "vc_hard_results.csv", index=False, header=True)
