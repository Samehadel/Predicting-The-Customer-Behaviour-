from sklearn.metrics import f1_score, accuracy_score
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import learning_curve, cross_val_score, GridSearchCV

def measure_accuracy(estimators, X, y):
    
    for e in estimators:
        e.fit(X, y)
        pred = e.predict(X)
        f1 = f1_score(y, pred)
        acc = accuracy_score(y, pred)
        print("{}: f1_score: {}, Accuracy_score: {} \n".format(e.__class__.__name__, f1, acc))


def plot_learning_curve(estimators, X, y, train_sizes, scorer, cv):
    
    #calculate required number of rows in the figure 
    n_rows = np.ceil(len(estimators) / 2)
    
    #calculate the width of the figure
    y_length = n_rows * 5 + 5
    
    # Create the figure window
    fig = plt.figure(figsize=(10, y_length))
    
    for i, est in enumerate(estimators):
        sizes, train_scores, test_scores = learning_curve(est, X, y, 
                                                          cv = cv, train_sizes = train_sizes, scoring = scorer)
        
        #print the done precentage
        print("Precentage of work done: {}%".format((i + 1) * 100 / len(estimators)))
        
        #get estimator name for title setting
        est_name = est.__class__.__name__
        
        # average train_scores and test_scores
        train_mean = np.mean(train_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)
        
        #Create subplots
        ax = fig.add_subplot(n_rows, 2, i + 1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        
        #add texts 
        ax.set_title(est_name)
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
       
    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 1.8), loc='lower left', borderaxespad = 0.)
    fig.suptitle('Learning Performances for Multiple Models', fontsize = 16, y = 1.03)
    fig.show()

def multi_grid_search(estimators, X, y, params, cv, scoring):
    
    grids = []
    
    for i, est in enumerate(estimators):
        
        #Define the grid search object
        grid_obj = GridSearchCV(est, param_grid = params[i], cv = cv, scoring = scoring)
        grid_obj.fit(X, y)
        grids.append(grid_obj)
        #print the done precentage
        print("Precentage of work done: {}%".format((i + 1) * 100 / len(estimators)))
   
    #return grid_obj
    return grids

def multi_cross_val(estimators, X, y, cv, scoring):
    
    scores = []
    
    for est in estimators:
        S = cross_val_score(est, X, y, cv =cv, scoring = scoring)
        scores.append(S)
        
    return scores

def cal_confusion_matrix(y_true, pred):
    POS = 0
    true_pos = 0
    
    NEG = 0
    true_neg = 0
    for i, element in enumerate(y_true):

        if element == 1:
            POS += 1
            if pred[i] == 1:
                true_pos += 1
        else:
            NEG += 1
            if pred[i] == 0 :
                true_neg += 1

    false_neg = POS - true_pos
    false_pos = NEG - true_neg
    
    return ['True Positive', 'False Positive', 'False Negative','True Negative'], [true_pos, false_pos, false_neg, true_neg]
    