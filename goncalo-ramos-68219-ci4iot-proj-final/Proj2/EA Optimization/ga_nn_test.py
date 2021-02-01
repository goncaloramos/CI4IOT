from sklearn import model_selection
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

class HyperparameterTuningGenetic:

    NUM_FOLDS = 5

    def __init__(self, randomSeed):

        self.randomSeed = randomSeed
        self.initDataset()
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, random_state=self.randomSeed)

    def initDataset(self):

        self.data = pd.read_csv('lab6-7-8_IoTGatewayCrash.csv')
        self.data_test = pd.read_csv('Proj2_IoTGatewayCrash.csv')

        self.data['PrevReq'] = self.data['Requests'].shift()
        self.data['PrevPrevReq'] = self.data['PrevReq'].shift()
        self.data = self.data.fillna(0)
        self.data['AvgHour'] = self.data[['Requests', 'PrevReq', 'PrevPrevReq']].mean(axis=1)
        self.cols = [self.col for self.col in self.data.columns if self.col not in ['Falha']]

        self.data_test['PrevReq'] = self.data_test['Requests'].shift()
        self.data_test['PrevPrevReq'] = self.data_test['PrevReq'].shift()
        self.data_test = self.data_test.fillna(0)
        self.data_test['AvgHour'] = self.data_test[['Requests', 'PrevReq', 'PrevPrevReq']].mean(axis=1)

        self.X = self.data[self.cols]
        self.y = self.data['Falha']

        self.X , self.y = self.overfit(self.X, self.y)

        self.X_new = self.data_test[self.cols]
        self.y_new = self.data_test['Falha']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)


    def create_tuple(self, neurons_layer1, neurons_layer2):
        if neurons_layer2 != 0:
            res=(neurons_layer1, neurons_layer2)
            return res
        else:
            res=(neurons_layer1,)
            return res

    def overfit(self,X, y):
        sm = SMOTE(sampling_strategy='all')
        X, y = sm.fit_sample(X, y)
        return X, y

    def convertParams(self, params):
        hidden_layer_sizes = self.create_tuple(round(params[0]), round(params[1]))
        activation = ['identity', 'logistic', 'tanh', 'relu'][round(params[2])]
        solver = ['lbfgs', 'sgd', 'adam'][round(params[3])]
        alpha = [0.0001, 0.0005][round(params[4])]
        learning_rate = ['constant', 'invscaling', 'adaptive'][round(params[5])]
        labels = []
        if ((round(params[6]) == 0) and (round(params[7]) == 0) and (round(params[8]) == 0) and (round(params[9]) == 0) and (round(params[10]) == 0)):
            labels.append('Load')
            labels.append('Requests')
            labels.append('PrevReq')
            labels.append('PrevPrevReq')
            labels.append('AvgHour')
        if (round(params[6]) == 1):
            labels.append('Load')
        if (round(params[7]) == 1):
            labels.append('Requests')
        if (round(params[8]) == 1):
            labels.append('PrevReq')
        if (round(params[9]) == 1):
            labels.append('PrevPrevReq')
        if (round(params[10]) == 1):
            labels.append('AvgHour')


        #return hidden_layer_sizes, activation, solver, alpha, learning_rate, num_input, combinations
        return hidden_layer_sizes, activation, solver, alpha, learning_rate, labels

    def getAccuracy(self, params):
        hidden_layer_sizes, activation, solver, alpha, learning_rate, labels = self.convertParams(params)
        self.classifier = MLPClassifier(random_state=self.randomSeed,
                                             hidden_layer_sizes=hidden_layer_sizes,
                                             activation=activation,
                                             solver=solver,
                                             alpha=alpha,
                                             learning_rate=learning_rate,
                                             max_iter = 1000
                                             )
        print(self.X)
        self.X_random = self.X[labels]

        cv_results = model_selection.cross_val_score(self.classifier,
                                                     self.X_random,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='accuracy')
        return cv_results.mean()



    def getFeatures(self, params):
        hidden_layer_sizes, activation, solver, alpha, learning_rate, labels = self.convertParams(params)

        return labels

    def formatParams(self, params):
        return "'hidden_layer-sizes'=%s,'activation'=%s, 'solver'=%solver, 'alpha'=%s, 'learning_rate'=%s, 'num_input'=%s"  % (self.convertParams(params))

    def bestNeuralNetworkFound(self, cols, params):

        print("Neural Network with best parametrization")

        hidden_layer_sizes = params[0]
        activation = params[1]
        solver = params[2]
        alpha = params[3]
        learning_rate = params[4]

        self.clf = MLPClassifier(random_state=self.randomSeed,
                                        hidden_layer_sizes=hidden_layer_sizes,
                                        activation=activation,
                                        solver=solver,
                                        alpha=alpha,
                                        learning_rate=learning_rate,
                                        max_iter=1000
                                        )

        self.cols_found = cols

        print(self.cols_found)

        self.X_train = self.X_train[self.cols_found]
        self.y_train = self.y_train
        self.X_test = self.X_test[self.cols_found]
        self.y_test = self.y_test
        self.X_new = self.X_new[self.cols_found]
        self.y_new = self.y_new

        self.clf.fit(self.X_train, self.y_train)

        # info
        print("Predicting Win/Loss on the test set using MultiLayer Perceptron")
        clf_pred = self.clf.predict(self.X_test)

        acc_score = round(accuracy_score(self.y_test, clf_pred), 4)
        print("Accuracy : ", acc_score)
        print("Classification Report: \n")
        print(classification_report(self.y_test, clf_pred))
        print("Confusion Matrix: \n")
        print(confusion_matrix(self.y_test, clf_pred))

        # info
        print("Predicting Win/Loss on the test set using MultiLayer Perceptron - New Test Dataset")
        clf_pred = self.clf.predict(self.X_new)

        acc_score = round(accuracy_score(self.y_new, clf_pred), 4)
        print("Accuracy : ", acc_score)
        print("Classification Report: \n")
        print(classification_report(self.y_new, clf_pred))
        print("Confusion Matrix: \n")
        print(confusion_matrix(self.y_new, clf_pred))