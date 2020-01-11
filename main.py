import numpy as np
import pandas as pd
# importujemy klasyfikatory
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# importujemy wszyskie metody selekcji cech
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, GenericUnivariateSelect

# metryki
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import binarize
import openml
from sklearn import datasets

#DATASET_PATH = 'datasets/Iris.csv'
#FEATURES = 5

def get_values_and_labels(ds):
#    return ds.iloc[:, :4], ds['Species']
    return ds.data, ds.target


def predictGenericUnivariateSelect(X, y, clf):
    features = GenericUnivariateSelect(chi2)
    features.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

    fit_chi2 = clf.fit(X_train, y_train)
    y_pred = fit_chi2.predict(X_test)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    f1_score = metrics.f1_score(y_test, y_pred, average=None)
    precision = metrics.precision_score(y_test, y_pred, average=None)
    recall = metrics.recall_score(y_test, y_pred, average=None)
    f1_scores.append(f1_score)
    precision_scores.append(precision)
    recall_scores.append(recall)

    return np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores)


def predictMutual(X, y, clf, k):
    features = SelectKBest(mutual_info_classif, k)
    features.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

    fit_chi2 = clf.fit(X_train, y_train)
    y_pred = fit_chi2.predict(X_test)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    f1_score = metrics.f1_score(y_test, y_pred, average=None)
    precision = metrics.precision_score(y_test, y_pred, average=None)
    recall = metrics.recall_score(y_test, y_pred, average=None)
    f1_scores.append(f1_score)
    precision_scores.append(precision)
    recall_scores.append(recall)

    return np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores)


def predictFCLassif(X, y, clf, k):
    features = SelectKBest(f_classif, k)
    features.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

    fit_chi2 = clf.fit(X_train, y_train)
    y_pred = fit_chi2.predict(X_test)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    f1_score = metrics.f1_score(y_test, y_pred, average=None)
    precision = metrics.precision_score(y_test, y_pred, average=None)
    recall = metrics.recall_score(y_test, y_pred, average=None)
    f1_scores.append(f1_score)
    precision_scores.append(precision)
    recall_scores.append(recall)

    return np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores)


def predictChi2(X, y, clf, k):
    features = SelectKBest(chi2, k)
    features.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

    fit_chi2 = clf.fit(X_train, y_train)
    y_pred = fit_chi2.predict(X_test)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    f1_score = metrics.f1_score(y_test, y_pred, average=None)
    precision = metrics.precision_score(y_test, y_pred, average=None)
    recall = metrics.recall_score(y_test, y_pred, average=None)
    f1_scores.append(f1_score)
    precision_scores.append(precision)
    recall_scores.append(recall)

    return np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores)


def predictRFE(X, y, clf):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

    rfe = RFE(clf, 4)
    fit_RFE = rfe.fit(X_train, y_train)
    y_pred = fit_RFE.predict(X_test)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    f1_score = metrics.f1_score(y_test, y_pred, average=None)
    precision = metrics.precision_score(y_test, y_pred, average=None)
    recall = metrics.recall_score(y_test, y_pred, average=None)
    f1_scores.append(f1_score)
    precision_scores.append(precision)
    recall_scores.append(recall)

    return np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores)


def main():
    svc = SVC()
    bayes = GaussianNB()
    tree = DecisionTreeClassifier()
    k=2
    #dataset = pd.read_csv(DATASET_PATH, encoding='latin-1')
    #X, y = get_values_and_labels(dataset)
    datas = ['analcatdata_michiganacc', 'analcatdata_seropositive',
             'newton_hema', 'vinnie', 'disclosure_z', 'arsenic-female-bladder', 'arsenic-male-lung', 'arsenic-female-lung', 'arsenic-male-bladder',
             'analcatdata_vineyard', 'rmftsa_sleepdata', 'visualizing_environmental', 'chscase_funds', 'witmer_census_1980', 'disclosure_z', 'chscase_geyser1', 'transplant',
                'hayes-roth', 'iris']
    # visualizing_livestock, chscase_geyser1, vinnie, chscase_vine2 arsenic-female-bladder analcatdata_neavote, balance-scale, analcatdata_challenger rmftsa_ctoarrivals
    #fruitfly analcatdata_dmft analcatdata_neavote analcatdata_challenger
    for i in datas:
        if (i=='hayes-roth' or i=='iris'):
            dataset = fetch_openml(name=i, version=1, cache=False)
        else:
            dataset = fetch_openml(name=i, version=2, cache=False)
        X, y = get_values_and_labels(dataset)
        '''print(X.shape)
        print(y.shape)'''
        #X = dataset.data
        #y = dataset.target
        #print(X)
        #print(y)

        resultRFE = [predictRFE(X, y, svc)]#, predictRFE(X, y, svc), predictRFE(X, y, tree)]
        #resultChi2 = [predictChi2(X, y, svc, k), predictChi2(X, y, bayes, k), predictChi2(X, y, tree, k)]
        #resultFCLassif = [predictFCLassif(X, y, svc, k), predictFCLassif(X, y, bayes, k), predictFCLassif(X, y, tree, k)]
        #resultMutual = [predictMutual(X, y, svc, k), predictMutual(X, y, bayes, k), predictMutual(X, y, tree, k)]
        #resultGenericUnivariateSelect = [predictGenericUnivariateSelect(X, y, svc), predictGenericUnivariateSelect(X, y, bayes), predictGenericUnivariateSelect(X, y, tree)]


        for i in resultRFE:
            print('{:2f}'.format(i[0]))#, '{:2f}'.format(i[1]), '{:2f}'.format(i[2]))
'''
        print('F1 Score: Precision: Recall: Chi2')
        for i in resultChi2:
            print('{:2f}'.format(i[0]), '{:2f}'.format(i[1]), '{:2f}'.format(i[2]))

        print('F1 Score: Precision: Recall: F_classif')
        for i in resultFCLassif:
            print('{:2f}'.format(i[0]), '{:2f}'.format(i[1]), '{:2f}'.format(i[2]))

        print('F1 Score: Precision: Recall: Mutual')
        for i in resultMutual:
            print('{:2f}'.format(i[0]), '{:2f}'.format(i[1]), '{:2f}'.format(i[2]))

        print('F1 Score: Precision: Recall: Generic')
        for i in resultGenericUnivariateSelect:
            print('{:2f}'.format(i[0]), '{:2f}'.format(i[1]), '{:2f}'.format(i[2]))
'''
        #el = [FEATURES]
        #el.extend(resultRFE)
        #print('{},{},{},{}'.format(*el))

if __name__ == '__main__':
    main()
