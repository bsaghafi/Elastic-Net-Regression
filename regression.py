# Inputs: targets and predictors
# Outputs: 5-fold cross validation train and test score and feature importance
# Date: 12/15/2017
# Author: Behrouz Saghafi
import operator
import random
import pandas as pd
from sklearn.preprocessing import Imputer, MinMaxScaler,StandardScaler
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score, permutation_test_score, KFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import timeit
import os
import nibabel as nib
from nilearn.image import resample_to_img
import fnmatch
import warnings
from sklearn.linear_model import ElasticNetCV,ElasticNet
warnings.filterwarnings("ignore")

# Set up possible values of parameters to optimize over
p_grid = {
    "l1_ratio": [.1, .5, .7, .9, .95, .99, 1],  # original one
    # "l1_ratio": [1],
    "alpha": float(2) ** np.arange(-10, 11)  # original one
    # "alpha": float(2)**np.arange(0,11)
}
start=timeit.default_timer()
scores=pd.read_csv('scores.csv')
aal=pd.read_csv('regional_values.csv')

combined_scores_aal = pd.merge(scores,aal, how='inner', on='Pal_ID')
combined_scores_aal=combined_scores_aal.dropna(subset=['reasoning_zscore'])
combined_scores_aal.to_csv('combined_scores_aal.csv')

# Use all features:
roi=np.load('All_roi.npy')
df=pd.read_csv('All_ROIs.csv')

image=nib.load('anatomical.nii.gz')
I=image.get_data()
I[::]=0

Atlas=np.zeros(I.shape)
pathname='MNI333/'

MNI='MNI_T1_orig.nii'
template=nib.load(MNI)

group_names=['old','middle_age','young','very_old']
for i in range(len(group_names)):
    group=group_names[i]
    print 'processing group of ',group
    X=combined_scores_aal.loc[combined_scores_aal['Group'] == group].iloc[:,22:].values
    y=combined_scores_aal.loc[combined_scores_aal['Group'] == group].iloc[:,21].values
    age=combined_scores_aal.loc[combined_scores_aal['Group'] == group].iloc[:,3].values
    age=np.reshape(age,(-1,1))
    X=np.concatenate((X,age),axis=1)
    roi=np.concatenate((roi,[265]))


    #compute z-scores:
    mean=np.mean(y)
    std=np.std(y)
    y=(y-mean)/std

    # X = Imputer(missing_values='NaN', strategy='mean', axis=1, verbose=0, copy=True).fit_transform(X)
    X=StandardScaler().fit_transform(X)
    # X=MinMaxScaler(feature_range=(-1,1)).fit_transform(X)

    models=[]
    
    ###################################################################################################### Nested & Non-Nested:
    # Number of random trials
    NUM_TRIALS = 1

    # Select Regressor or Classifier:
    eNet = ElasticNet(max_iter=10000)

    # Arrays to store scores
    non_nested_scores = np.zeros(NUM_TRIALS)
    train_nn_scores=np.zeros(NUM_TRIALS)
    nested_scores = np.zeros(NUM_TRIALS)
    train_scores = np.zeros(NUM_TRIALS)

    best_params=[]

    # Loop for each trial
    for i in range(NUM_TRIALS):

        # print 'Trial #',i+1

        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = GridSearchCV(estimator=eNet, param_grid=p_grid, cv=inner_cv, n_jobs=-1)
        clf.fit(X, y)
        # non_nested_scores[i] = clf.best_score_
        best_params.append(clf.best_params_)
        estimator=clf.best_estimator_
        print 'best params:',clf.best_params_
        print 'total coefficients=', len(estimator.coef_)
        ####feature selection:
        # print 'coefficients=', estimator.coef_
        fcount=np.count_nonzero(estimator.coef_)
        print '# of features selected=',fcount
        # print 'features selected: seed numbers', np.nonzero(estimator.coef_)
        weights = np.array(estimator.coef_)
        weights_abs = np.absolute(weights)
        weights_sorted = np.sort(weights_abs)
        indices = np.argsort(weights_abs)
        colors = []
        for f in range(fcount):
            if weights[indices[-f-1]] > 0:
                colors.append('lawngreen')
            else:
                colors.append('deepskyblue')
        index_selected=indices[-1:-fcount - 1:-1]
        # print 'features selected: seed numbers', roi[index_selected]
        name=group+'_Fselected'
        np.save(name,roi[index_selected])
        name = group + '_Weights'
        np.save(name, weights_abs[index_selected])
        fig1, ax = plt.subplots()
        # print 'weights=', weights[index_selected]
        ax.bar(range(fcount), weights_abs[index_selected], color=colors)
        ax.set_xticks(np.arange(0, fcount, 1))
        # ax.set_xticklabels(roi[index_selected], rotation='vertical')
        ax.set_xticklabels(df.iloc[index_selected,15].values, rotation='vertical')  #Gyri=7, AAL=10
        ax.set_title(group)
        ax.set_xlabel('predictor')
        ax.set_ylabel('absolute weight')
        g_patch = mpatches.Patch(color='lawngreen', label='Positive')
        b_patch = mpatches.Patch(color='deepskyblue', label='Negative')
        plt.legend(handles=[g_patch,b_patch])
        # plt.tight_layout()
        plt.show()
        ###########
        # cv_results:
        test_cv_results=[]
        train_cv_results=[]
        for train_index, test_index in inner_cv.split(X, y):
            estimator.fit(X[train_index], y[train_index])
            test_cv_results.append(estimator.score(X[test_index], y[test_index]))
            train_cv_results.append(estimator.score(X[train_index], y[train_index]))
        non_nested_scores[i]=np.array(test_cv_results).mean()
        train_nn_scores[i] = np.array(train_cv_results).mean()
        [score,permutation_scores,pvalue]=permutation_test_score(estimator,X,y,cv=inner_cv, n_jobs=-1, n_permutations=10000)
        print 'pvalue=', pvalue

        # Nested CV with parameter optimization
        # nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
        # nested_scores[i] = nested_score.mean()
        # test_cv_results=[]
        # train_cv_results=[]
        # for train_index, test_index in outer_cv.split(X, y):
        #     clf.fit(X[train_index], y[train_index])
        #     test_cv_results.append(clf.score(X[test_index], y[test_index]))
        #     train_cv_results.append(clf.score(X[train_index], y[train_index]))
        # nested_scores[i]=np.array(test_cv_results).mean()
        # train_scores[i] = np.array(train_cv_results).mean()

    print 'non-nested cross-validation'
    msg = "%s: %f" % ('test score', non_nested_scores.mean())
    print(msg)
    msg = "%s: %f" % ('train score', train_nn_scores.mean())
    print(msg)

