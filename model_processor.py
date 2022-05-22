"""
Contain functionalities use to fit, predict and evaluate ML models
"""
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

# Function that runs the requested algorithms and return the accuracy matrics
def fit_ml_algo(algo, X_train, y_train, cv):
    """
    Fit data to the algorthm and make accurcy scores
    :param algo: algorithm
    :param X_train: train set
    :param y_train: target set
    :param cv: number of cross validations
    :return: predictions, model accuracy, cross validation score
    """
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)

    # Cross validation
    train_pred = cross_val_predict(algo, X_train, y_train, cv=cv, n_jobs=-1)

    # Cross-validation accuracy metrics
    acc_cv = round(accuracy_score(y_train, train_pred) * 100, 2)

    return train_pred, acc, acc_cv

def show_performance(model, algo_name):
    """
    Indicate the performance of the fine-tuned model
    :param classifier: fine-tuned model
    :param model_name: algorith name
    :return:
    """
    print(algo_name)
    print('Best Score:\t', str(model.best_score_))
    print('Best Parameters:\t ' + str(model.best_params_))
