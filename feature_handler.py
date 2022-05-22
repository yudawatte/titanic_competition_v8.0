"""
Contain functionalities use for feature engineering
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from model_processor import show_performance
import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def predict_missing_age_values(dataset):
    """
    Use to fill missing age values base on features 'Pclass', 'SibSp', and 'Parch'
    :param dataset: dataset (train set + test set)
    :return: data set with filled missing age values
    """

    # Take not null Age values
    dataset_with_age = dataset[dataset.Age.notnull()]

    # Prepare dataset to fit to a prediction model
    X_train = dataset_with_age[["Pclass", "SibSp", "Parch"]]
    y_train = dataset_with_age["Age"]

    X_test = dataset[["Pclass", "SibSp", "Parch"]]

    # Train a random foreset regressor model
    rf_reg = RandomForestRegressor(random_state=2,
                                   bootstrap=False,
                                   max_depth=7,
                                   max_features='sqrt',
                                   min_samples_leaf=2,
                                   min_samples_split=3,
                                   n_estimators=5).fit(X_train, y_train)

    # Predict Age values
    predict = rf_reg.predict(X_test)
    dataset["Age_new"] = predict
    index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

    for i in index_NaN_age:
        dataset["Age"].iloc[i] = dataset.iloc[i]["Age_new"]

    # Drop variable: Age_new
    dataset.drop(labels=["Age_new"], axis=1, inplace=True)

    return dataset


def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices corresponding to the observations containing more than
    n outliers according to the Tukey's method
    :param df: incoming dataset
    :param n: observation count threshold
    :param features:
    :return: lis of outlier indecies
    """
    outlier_indices = []

    # Iterate over feature
    for col in features:
        # 1st quartile
        Q1 = np.percentile(df[col], 25)

        # 3rd quartile
        Q3 = np.percentile(df[col], 75)

        # Inter quartile range(IQR)
        IQR = Q3 - Q1

        # Outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # Determine a list of indices for feature to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # Select observations containing more that n outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers

def detect_feature_importance(name, classifier, X_train):
    """

    :param model:
    :return:
    """
    nrows = ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex="all", figsize=(15, 15))
    #fig, axes = plt.plot()
    #names_classifiers = [("AdaBoosting", ada_best), ("ExtraTrees", ExtC_best), ("RandomForest", RFC_best),
    #                     ("GradientBoosting", GBC_best)]

    #nclassifier = 0
    for row in range(nrows):
        for col in range(ncols):
            #name = names_classifiers[nclassifier][0]
            #classifier = names_classifiers[nclassifier][1]
            indices = np.argsort(classifier.feature_importances_)[::-1][:80]
            print(indices)
            g = sns.barplot(y=X_train.columns[indices][:80], x=classifier.feature_importances_[indices][:80],
                            orient='h', ax=axes[row][col])
            g.set_xlabel("Relative importance", fontsize=12)
            g.set_ylabel("Features", fontsize=12)
            g.tick_params(labelsize=9)
            g.set_title(name + " feature importance")
            #nclassifier += 1

    #fig.save("test")
    #fig.savefig('test.png')