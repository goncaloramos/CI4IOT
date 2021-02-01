import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn import model_selection



def duplicate_crash(df, n):
    print("len train", len(df))

    reps = [n if val == 1 else 1 for val in df.Falha]
    new_train = df.loc[np.repeat(df.index.values, reps)]

    print("len new train", len(new_train))

    return new_train


def add_new_features(df):
    df['PrevReq'] = df['Requests'].shift()
    df['PrevPrevReq'] = df['PrevReq'].shift()
    df = df.fillna(0)
    print(df.head())
    return df


def add_fuzzy_logic(df):
    new_df = df
    new_df.loc[(new_df.PrevReq >= 0.4) & (new_df.PrevReq < 0.7), 'PrevReq'] = 0.5
    new_df.loc[new_df.PrevReq < 0.4, 'PrevReq'] = 0
    new_df.loc[new_df.PrevReq >= 0.7, 'PrevReq'] = 1
    new_df.loc[(new_df.PrevPrevReq >= 0.4) & (new_df.PrevPrevReq < 0.7), 'PrevPrevReq'] = 0.5
    new_df.loc[new_df.PrevPrevReq < 0.4, 'PrevPrevReq'] = 0
    new_df.loc[new_df.PrevPrevReq >= 0.7, 'PrevPrevReq'] = 1
    new_df.loc[new_df.Load >= 0.6, 'Load'] = 1
    new_df.loc[new_df.Load < 0.4, 'Load'] = 0
    new_df.loc[(new_df.Load >= 0.4) & (new_df.Load < 0.6), 'PrevPrevReq'] = 0.5
    print("fuzzy-logic:", new_df.head())
    return new_df

def encoding(df):
    # create the labelEncoder object
    le = preprocessing.LabelEncoder()
    # convert the categorical columns into numeric
    df['PrevReq'] = le.fit_transform(df['PrevReq'])
    df['PrevPrevReq'] = le.fit_transform(df['PrevPrevReq'])
    df['Load'] = le.fit_transform(df['Load'])
    return df


def overfit(X,y):
    sm = SMOTE(sampling_strategy = 'all')
    X, y = sm.fit_sample(X,y)
    return X, y


def naive_bayes(X_train, y_train, X_test, y_test):

    # create an object of the type GaussianNB
    gnb = GaussianNB()

    # train the algorithm on training data and predict using the testing data
    pred = gnb.fit(X_train, y_train).predict(X_test)
    print("GaussianNB Pred:\n", pred.tolist())

    # print the accuracy score of the model
    acc_score = round(accuracy_score(y_test, pred), 4)
    print("GaussianNB accuracy_score:\n", acc_score)

    # visualizer = ClassificationReport(gnb, classes=['Won', 'Loss'])
    # visualizer.fit(X_train, y_train)
    # visualizer.score(X_test, y_test)
    # g = visualizer.poof()

    return acc_score


def mlp_classifier(X_train, y_train, X_test, y_test):
    print("Running MLP Classifier..")
    # Choosing Classifier
    clf = MLPClassifier()

    # Defining a hyper-parameter space to search
    parameter_space = {
        'hidden_layer_sizes': [(2,), (2, 1), (3,), (3, 1)],
        'activation': ['tanh'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.0005],
        'learning_rate': ['constant', 'adaptive'],
    }

    # Run the search
    clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=2) #cv = 2

    clf.fit(X_train, y_train)

    # info
    print('Best parameters found:\n', clf.best_params_)
    print("Predicting Win/Loss on the test set using MultiLayer Perceptron")
    clf_pred = clf.predict(X_test)

    acc_score = round(accuracy_score(y_test, clf_pred), 4)
    print("Accuracy : ", acc_score)
    print("Classification Report: \n")
    print(classification_report(y_test, clf_pred))
    print("Confusion Matrix: \n")
    print(confusion_matrix(y_test, clf_pred))

    # plot confusion matrix
    #skplt.metrics.plot_confusion_matrix(y_test, clf_pred)
    #plt.show()

    return acc_score


def neural_network_past():
    print("Starting IoTGatewayCrash..\n")
    random.seed(42)

    df = pd.read_csv('lab6-7-8_IoTGatewayCrash.csv')
    df_test = pd.read_csv('Proj2_IoTGatewayCrash.csv')

    #le = preprocessing.LabelEncoder()
    #le.fit(["Normal Operation", "Crash"])

    #print("DF Classes:\n", list(le.classes_))

    df = add_new_features(df)
    df = add_fuzzy_logic(df)
    df = encoding(df)

    df_test = add_new_features(df_test)
    df_test = add_fuzzy_logic(df_test)
    df_test = encoding(df_test)

    train, val, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

    new_train = duplicate_crash(train, 3)

    cols = [col for col in df.columns if col not in ['Falha', 'Requests']]

    print("cols", cols)

    # set features variables
    data_train = train[cols]
   # data_train = new_train[cols]
    data_val = val[cols]
    data_test = test[cols]

    # set target variable
    target_train = train['Falha']


    #target_train = new_train['Falha']
    target_val = val['Falha']
    target_test = test['Falha']

    data_new_test = df_test[cols]
    target_new_test = df_test['Falha']


    # apply standard scaling to get optimized result
    sc = StandardScaler()
    data_train = sc.fit_transform(data_train)
    data_test = sc.fit_transform(data_test)
    data_val = sc.fit_transform(data_val)
    data_new_test = sc.fit_transform(data_new_test)
    print("data_train_sc:", data_train)
    print("data_val_sc:", data_val)
    print("data_test_sc:", data_test)

    #data_train, target_train = overfit(data_train, target_train)


    # Run MLPC
    classifier_acc = mlp_classifier(data_train, target_train, data_test, target_test)
    print("NEW DATASET\n")
    classifier_acc = mlp_classifier(data_train, target_train, data_new_test, target_new_test)

    # Compare GaussianNB and Classifier accuracy
    print("MLP Classifier accuracy:", classifier_acc)


neural_network_past()