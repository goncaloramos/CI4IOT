import random

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import skfuzzy as fuzz
from skfuzzy import control as ctrl


def add_new_features(df):
    df['PrevReq'] = df['Requests'].shift()
    df['PrevPrevReq'] = df['PrevReq'].shift()
    df = df.fillna(0)
    print("DataFrame head: \n")
    print(df.head())
    return df


def encoding(df):
    # create the labelEncoder object
    le = preprocessing.LabelEncoder()
    # convert the categorical columns into numeric
    df['PrevReq'] = le.fit_transform(df['PrevReq'])
    df['PrevPrevReq'] = le.fit_transform(df['PrevPrevReq'])
    df['Load'] = le.fit_transform(df['Load'])
    return df


def fuzzy_system(pv, ppv, ld):

    prev_req = ctrl.Antecedent(np.arange(0, 1.001, 0.001), 'prev_requests')
    prevprev_req = ctrl.Antecedent(np.arange(0, 1.001, 0.001), 'prevprev_requests')
    load = ctrl.Antecedent(np.arange(0, 1.001, 0.001), 'load')
    crash = ctrl.Consequent(np.arange(0, 1.001, 0.001), 'crash')

    prev_req['poor'] = fuzz.trimf(prev_req.universe, [0, 0, 0.4])
    prev_req['average'] = fuzz.trimf(prev_req.universe, [0, 0.4, 0.6])
    prev_req['good'] = fuzz.trimf(prev_req.universe, [0.6, 1, 1])

    prevprev_req['poor'] = fuzz.trimf(prevprev_req.universe, [0, 0, 0.4])
    prevprev_req['average'] = fuzz.trimf(prevprev_req.universe, [0, 0.4, 0.6])
    prevprev_req['good'] = fuzz.trimf(prevprev_req.universe, [0.6, 1, 1])

    load['poor'] = fuzz.trimf(load.universe, [0, 0, 0.4])
    load['average'] = fuzz.trimf(load.universe, [0, 0.4, 0.6])
    load['good'] = fuzz.trimf(load.universe, [0.6, 1, 1])


    ##########################################################
    #                      Membership Definition             #
    ##########################################################

    prev_req.automf(3)
    prevprev_req.automf(3)
    load.automf(3)

    crash['NO'] = fuzz.trimf(crash.universe, [0,0,0.4])
    crash['Crash'] = fuzz.trimf(crash.universe, [0.4,1,1])

    #prev_req.view()
    #prevprev_req.view()
    #load.view()
    #crash.view()
    #plt.show()

    ##########################################################
    #                      Rules Definition                  #
    ##########################################################

    rule1 = ctrl.Rule(prev_req['poor'] & prevprev_req['poor'] & load['poor'], crash['NO'])
    rule2 = ctrl.Rule(prev_req['poor'] & prevprev_req['average'] & load['poor'], crash['NO'])
    rule3 = ctrl.Rule(prev_req['poor'] & prevprev_req['good'] & load['poor'], crash['NO'])

    rule4 = ctrl.Rule(prev_req['average'] & prevprev_req['poor'] & load['poor'], crash['NO'])
    rule5 = ctrl.Rule(prev_req['average'] & prevprev_req['average'] & load['poor'], crash['NO'])
    rule6 = ctrl.Rule(prev_req['average'] & prevprev_req['good'] & load['poor'], crash['NO'])

    rule7 = ctrl.Rule(prev_req['good'] & prevprev_req['poor'] & load['poor'], crash['NO'])
    rule8 = ctrl.Rule(prev_req['good'] & prevprev_req['average'] & load['poor'], crash['NO'])
    rule9 = ctrl.Rule(prev_req['good'] & prevprev_req['good'] & load['poor'], crash['NO'])

    rule10 = ctrl.Rule(prev_req['poor'] & prevprev_req['poor'] & load['average'], crash['NO'])
    rule11 = ctrl.Rule(prev_req['poor'] & prevprev_req['average'] & load['average'], crash['NO'])
    rule12 = ctrl.Rule(prev_req['poor'] & prevprev_req['good'] & load['average'], crash['NO'])

    rule13 = ctrl.Rule(prev_req['average'] & prevprev_req['poor'] & load['average'], crash['NO'])
    rule14 = ctrl.Rule(prev_req['average'] & prevprev_req['average'] & load['average'], crash['NO'])
    rule15 = ctrl.Rule(prev_req['average'] & prevprev_req['good'] & load['average'], crash['NO'])

    rule16 = ctrl.Rule(prev_req['good'] & prevprev_req['poor'] & load['average'], crash['NO'])
    rule17 = ctrl.Rule(prev_req['good'] & prevprev_req['average'] & load['average'], crash['NO'])
    rule18 = ctrl.Rule(prev_req['good'] & prevprev_req['good'] & load['average'], crash['NO'])

    rule19 = ctrl.Rule(prev_req['poor'] & prevprev_req['poor'] & load['good'], crash['NO'])
    rule20 = ctrl.Rule(prev_req['poor'] & prevprev_req['average'] & load['good'], crash['NO'])
    rule21 = ctrl.Rule(prev_req['poor'] & prevprev_req['good'] & load['good'], crash['NO']) #maybe

    rule22 = ctrl.Rule(prev_req['average'] & prevprev_req['poor'] & load['good'], crash['NO'])
    rule23 = ctrl.Rule(prev_req['average'] & prevprev_req['average'] & load['good'], crash['Crash'])
    rule24 = ctrl.Rule(prev_req['average'] & prevprev_req['good'] & load['good'], crash['Crash'])

    rule25 = ctrl.Rule(prev_req['good'] & prevprev_req['poor'] & load['good'], crash['Crash'])
    rule26 = ctrl.Rule(prev_req['good'] & prevprev_req['average'] & load['good'], crash['Crash'])
    rule27 = ctrl.Rule(prev_req['good'] & prevprev_req['good'] & load['good'], crash['Crash'])

    input_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27])

    crsh = ctrl.ControlSystemSimulation(input_control)


    ##########################################################
    #                      Inputs Definition                 #
    ##########################################################

    crsh.input['prev_requests'] = pv
    crsh.input['prevprev_requests'] = ppv
    crsh.input['load'] = ld

    print("inputs", pv, ppv, ld)

    crsh.compute()

    result = crsh.output['crash']
    print(result)
    #crash.view(sim=crsh)
    #plt.show()
    return result


def script():
    print("Starting IoTGatewayCrash..\n")
    random.seed(42)

    df = pd.read_csv('lab6-7-8_IoTGatewayCrash.csv')
    test_dataset = pd.read_csv('Proj2_IoTGatewayCrash.csv')

    df = add_new_features(df)
    test_dataset = add_new_features(test_dataset)

    target = df['Falha']
    target_test_dataset = test_dataset['Falha']

    print("target head: \n")
    print(target.head())

    arr = []

    for index, row in df.iterrows():
        pv_v = row['PrevReq']
        ppv_v = row['PrevPrevReq']
        ld_v = row['Load']

        res = int(round(fuzzy_system(pv_v, ppv_v, ld_v)))

        if(res == 1):
            print("Crash", pv_v, ppv_v, ld_v)

        arr.append(res)

    print("Results: \n")
    print(arr)
    print("System Accuracy: \n")
    print(round(accuracy_score(target, arr), 4))
    print("Evaluation Report: \n")
    print(classification_report(target, arr))
    print("Confusion Matrix: \n")
    print(confusion_matrix(target, arr))

    arrNewTest = []

    for index, row in test_dataset.iterrows():
        pv_v = row['PrevReq']
        ppv_v = row['PrevPrevReq']
        ld_v = row['Load']

        res = int(round(fuzzy_system(pv_v, ppv_v, ld_v)))

        if (res == 1):
            print("Crash", pv_v, ppv_v, ld_v)

        arrNewTest.append(res)

    print("Results: \n")
    print(arrNewTest)
    print("System Accuracy: \n")
    print(round(accuracy_score(target_test_dataset, arrNewTest), 4))
    print("Evaluation Report: \n")
    print(classification_report(target_test_dataset, arrNewTest))
    print("Confusion Matrix: \n")
    print(confusion_matrix(target_test_dataset, arrNewTest))




script()

5