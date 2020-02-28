# Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tqdm import trange

examDict = {'time': [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                     2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50],
            'score': [10, 22, 13, 43, 20, 22, 33, 50, 62, 48,
                   55, 75, 62, 73, 81, 76, 64, 82, 90, 93]}

examDf = pd.DataFrame(examDict)
train_percentage = 0.8
round = 100
k_max = 0
b_max = 0
score_max = 0
for i in trange(round):

    '''
    print(examDf)
    
    plt.scatter(examDf['time'], examDf['score'], color='c')
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.show()
    
    print(examDf.corr())
    '''

    X_train, X_test, Y_train, Y_test = train_test_split(examDf['time'], examDf['score'], train_size=train_percentage,
                                                        test_size=1-train_percentage)

    '''
    plt.scatter(X_train, Y_train, color="b", label="train data")
    plt.scatter(X_test, Y_test, color="r", label="test data")
    plt.legend(loc=4)
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.show()
    '''

    model = LinearRegression()
    # X_train.values is a NumPy Array
    X_train = X_train.values.reshape(-1, 1)
    Y_train = Y_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)
    Y_test = Y_test.values.reshape(-1, 1)

    model.fit(X_train, Y_train)

    k = model.coef_
    b = model.intercept_

    y_pred = model.predict(X_train)
    plt.plot(X_train, y_pred, color='black', linewidth=2, label="Fit Line")
    plt.scatter(X_train, Y_train, color='blue', label='Train Data')
    plt.scatter(X_test, Y_test, color='red', label="Test Data")

    plt.legend(loc=2)
    plt.xlabel("Time")
    plt.ylabel("Score")
    # plt.savefig('G:/pythonFiles/fig1.png',dpi=1000)
    # plt.show()

    score = model.score(X_test, Y_test)
    #print(score)   # coefficient of determination

    if score > score_max:
        score_max = score
        k_max = k
        b_max = b

    pass

print("Best Fitting: "+'y = ' + str(format(k_max[0, 0],'.4f')) + ' x + ' + str(format(b_max[0],'.4f')))
print('ScoreMax: ',score_max)
