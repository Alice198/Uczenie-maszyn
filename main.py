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
from sklearn.model_selection import train_test_split, cross_val_score


from sklearn.datasets import fetch_openml
from scipy.stats import wilcoxon, t

import sys
#DATASET_PATH = 'datasets/Iris.csv'
#FEATURES = 5

def get_values_and_labels(ds):
#    return ds.iloc[:, :4], ds['Species']
    return ds.data, ds.target




def predictGenericUnivariateSelect(X, y, clf):
    features = GenericUnivariateSelect(chi2)
    features.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    fit_chi2 = clf.fit(X_train, y_train)
    y_pred = fit_chi2.predict(X_test)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    f1_score = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.f1_score(y_test, y_pred, average=None)
    recall = metrics.recall_score(y_test, y_pred, average=None)
    f1_scores.append(f1_score)
    precision_scores.append(precision)
    recall_scores.append(recall)

    return np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores)


def predictMutual(X, y, clf, k):
    features = SelectKBest(mutual_info_classif, k)
    features.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    fit_chi2 = clf.fit(X_train, y_train)
    y_pred = fit_chi2.predict(X_test)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    f1_score = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.f1_score(y_test, y_pred, average=None)
    recall = metrics.recall_score(y_test, y_pred, average=None)
    f1_scores.append(f1_score)
    precision_scores.append(precision)
    recall_scores.append(recall)

    return np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores)


def predictFCLassif(X, y, clf, k):
    features = SelectKBest(f_classif, k)
    features.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    fit_chi2 = clf.fit(X_train, y_train)
    y_pred = fit_chi2.predict(X_test)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    f1_score = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.f1_score(y_test, y_pred, average=None)
    recall = metrics.recall_score(y_test, y_pred, average=None)
    f1_scores.append(f1_score)
    precision_scores.append(precision)
    recall_scores.append(recall)

    return np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores)


def predictChi2(X, y, clf, k):
    features = SelectKBest(chi2, k)
    a = features.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    fit_chi2 = clf.fit(X_train, y_train)
    y_pred = fit_chi2.predict(X_test)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    f1_score = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.f1_score(y_test, y_pred, average=None)
    recall = metrics.recall_score(y_test, y_pred, average=None)
    f1_scores.append(f1_score)
    precision_scores.append(precision)
    recall_scores.append(recall)

    return np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores)


def predictRFE(X, y, clf):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    rfe = RFE(clf, 4)
    fit_RFE = rfe.fit(X_train, y_train)
    y_pred = fit_RFE.predict(X_test)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    f1_score = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.f1_score(y_test, y_pred, average=None)
    recall = metrics.recall_score(y_test, y_pred, average=None)
    f1_scores.append(f1_score)
    precision_scores.append(precision)
    recall_scores.append(recall)

    return np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores)


def predictNotFeatures(X, y, clf):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
    fitNot = clf.fit(X_train, y_train)
    y_pred = fitNot.predict(X_test)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    f1_score = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.f1_score(y_test, y_pred, average=None)
    recall = metrics.recall_score(y_test, y_pred, average=None)
    f1_scores.append(f1_score)
    precision_scores.append(precision)
    recall_scores.append(recall)

    return np.mean(f1_scores), np.mean(precision_scores), np.mean(recall_scores)



def main():
    svc = SVC()
    bayes = GaussianNB()
    tree = DecisionTreeClassifier()

    classifier = [svc, bayes, tree]
    k=2
    #dataset = pd.read_csv(DATASET_PATH, encoding='latin-1')
    #X, y = get_values_and_labels(dataset)
    datas = ['analcatdata_michiganacc', 'analcatdata_seropositive','arsenic-female-bladder', 'arsenic-male-lung', 'arsenic-female-lung', 'arsenic-male-bladder',
             'analcatdata_vineyard', 'rmftsa_sleepdata', 'chscase_geyser1', 'transplant', 'hayes-roth', 'iris', 'visualizing_environmental', 'chscase_funds',
             'disclosure_z', 'disclosure_x_bias', 'servo', 'newton_hema', 'vinnie', 'diggle_table_a1']

    tabSVCChi2 = []
    tabSVCCLassif = []
    tabSVCGenericUnivariateSelect = []
    tabSVCMutual = []
    tabSVCRFE = []
    tabBayesChi2 = []
    tabBayesClassif = []
    tabBayesGenericUnivariateSelect = []
    tabBayesMutual = []
    tabBayesRFE  = []
    tabTreeChi2 = []
    tabTreeClassif = []
    tabTreeGenericUnivariateSelect = []
    tabTreeMutual = []
    tabTreeRFE = []
    for i in datas:
        if (i=='hayes-roth' or i=='iris' or i=='servo'):
            dataset = fetch_openml(name=i, version=1, cache=False)
        else:
            dataset = fetch_openml(name=i, version=2, cache=False)

        X, y = get_values_and_labels(dataset)

        #X = dataset.data
        #y = dataset.target
        #print(X)
        #print(y)

        resultSVC =[predictNotFeatures(X, y , svc), predictChi2(X, y, svc, k),  predictFCLassif(X, y, svc, k), predictGenericUnivariateSelect(X,y, svc), predictMutual(X,y , svc, k)]
        resultBayes = [predictNotFeatures(X, y , bayes), predictChi2(X, y, bayes, k),  predictFCLassif(X, y, bayes, k), predictGenericUnivariateSelect(X,y, bayes), predictMutual(X,y , bayes, k)]
        resultTree = [predictNotFeatures(X, y , tree), predictChi2(X, y, tree, k),  predictFCLassif(X, y, tree, k), predictGenericUnivariateSelect(X,y, tree), predictMutual(X,y , tree, k)]

       # for i in resultSVC:
       #     print('{:2f}'.format(i[0]), '{:2f}'.format(i[1]), '{:2f}'.format(i[2]), end=' ')
       # print()
        print(predictNotFeatures(X, y , svc))

        result1 = [resultSVC[0][0],resultSVC[1][0], resultSVC[2][0], resultSVC[3][0], resultSVC[4][0]]
        result2 = [resultBayes[0][0], resultBayes[1][0], resultBayes[2][0], resultBayes[3][0], resultBayes[4][0]]
        result3 = [resultTree[0][0], resultTree[1][0], resultTree[2][0], resultTree[3][0], resultTree[4][0]]

        SVCChi2 = resultSVC[0][0]
        tabSVCChi2.append(SVCChi2)
        SVCCLassif = resultSVC[1][0]
        tabSVCCLassif.append(SVCCLassif)
        SVCGenericUnivariateSelect = resultSVC[2][0]
        tabSVCGenericUnivariateSelect.append(SVCGenericUnivariateSelect)
        SVCMutual = resultSVC[3][0]
        tabSVCMutual.append(SVCMutual)
        SVCRFE = resultSVC[4][0]
        tabSVCRFE.append(SVCRFE)
        BayesChi2 = resultBayes[0][0]
        tabBayesChi2.append(BayesChi2)
        BayesClassif = resultBayes[1][0]
        tabBayesClassif.append(BayesClassif)
        BayesGenericUnivariateSelect = resultBayes[2][0]
        tabBayesGenericUnivariateSelect.append(BayesGenericUnivariateSelect)
        BayesMutual = resultBayes[3][0]
        tabBayesMutual.append(BayesMutual)
        BayesRFE = resultBayes[4][0]
        tabBayesRFE.append(BayesRFE)
        TreeChi2 = resultTree[0][0]
        tabTreeChi2.append(TreeChi2)
        TreeClassif = resultTree[1][0]
        tabTreeClassif.append(TreeClassif)
        TreeGenericUnivariateSelect = resultTree[2][0]
        tabTreeGenericUnivariateSelect.append(TreeGenericUnivariateSelect)
        TreeMutual = resultTree[3][0]
        tabTreeMutual.append(TreeMutual)
        TreeRFE = resultTree[4][0]
        tabTreeRFE.append(TreeRFE)

    tabSVC = [tabSVCChi2, tabSVCCLassif, tabSVCGenericUnivariateSelect, tabSVCMutual, tabSVCRFE]
    tabBayes = [tabBayesChi2, tabBayesClassif, tabBayesGenericUnivariateSelect, tabBayesMutual, tabBayesRFE]
    tabTree = [tabTreeChi2, tabTreeClassif, tabTreeGenericUnivariateSelect, tabTreeMutual, tabTreeRFE]

'''
    for s in range(0, 5):
        a, b = wilcoxon(tabSVC[s], tabBayes[s])
        print(a, b, end=' ')
    print()

    for c in range(0, 5):
        a, b = wilcoxon(tabSVC[c], tabTree[c])
        print(a, b, end=' ')
    print()

    for v in range(0, 5):
        a, b = wilcoxon(tabBayes[v], tabTree[v])
        print(a, b, end=' ')
    print()
'''


if __name__ == '__main__':
    main()
