# import relevant libraries
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, roc_curve, classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTETomek
import pickle
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot


def load_datasets():
    conn = sqlite3.connect(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\3. Source_Codes\databases\chronic_diseases.sqlite3")
    hypertension_df = pd.read_sql("SELECT bmi, dbp, sbp, outcome FROM hypertension", conn)
    
    # create a set of copy of diabetes dataset
    copy_hypertension_df = hypertension_df.copy()
    
    conn.close()
    return copy_hypertension_df

def missing_data(bp_df):
    plt.title("Investigate Missing Values in Hypertension Dataset")
    sns.heatmap(bp_df.isnull(), annot = False)
    plt.show()

def categorical_class_to_num(bp_df):
    lb = LabelBinarizer()
    bp_df["outcome"] = lb.fit_transform(bp_df["outcome"])
    return bp_df

def regression_analysis(bp_df):
    # generate a pairplot with regression plot to illustrate the relationship between variables
    sns.pairplot(bp_df, kind = "reg", hue = "outcome", palette = "coolwarm")
    plt.show()
    
    # generate a heatmap to illustrate the Pearson correlation between variables
    plt.figure(figsize = (15, 6))
    sns.heatmap(bp_df.corr(), annot = True, cmap = "Blues")
    plt.title("Heatmap of Hypertension", fontweight = 'bold', fontsize = 14)
    plt.tight_layout()
    plt.show()

def distribution_curve(bp_df, feature_names):
    for feature in feature_names:
        sns.displot(x = feature, data = bp_df, hue = "outcome", kde = True, palette = "coolwarm")
        plt.xlabel(feature, fontsize = 14, fontweight = "bold")
        plt.ylabel("Count", fontsize = 14, fontweight = "bold")
        plt.show()
        print()

def boxplot(bp_df):
    sns.boxplot(data = bp_df.drop('outcome', axis = 1), orient = 'h', palette = 'Set2')
    plt.show()

def histogram(bp_df):
    bp_df.hist()
    plt.tight_layout()
    pyplot.show()

def balanced_datasets(X, y):
    oversample_SMT = SMOTETomek(random_state=1)
    x_smt, y_smt = oversample_SMT.fit_resample(X, y)
    return x_smt, y_smt


def cross_validation_eval(pipe, X_smt, Y_smt):
    
    shuffle_split = ShuffleSplit(test_size = 0.20, train_size = 0.80, n_splits = 5, random_state = 0)
    cv_result = cross_validate(pipe, 
                               X_smt, 
                               Y_smt, 
                               cv = shuffle_split,
                               n_jobs = 1,
                               scoring = ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc'),
                               return_train_score = True,
                               return_estimator = True)
    return cv_result

def cross_validation_summary(model_names, cv_trainScores_acc, cv_testScores_acc, cv_trainScores_prec, cv_testScores_prec, cv_trainScores_recall, cv_testScores_recall, cv_trainScores_f1, cv_testScores_f1, cv_trainScores_roc, cv_testScores_roc):
    cv_evalSummary = pd.DataFrame({'Train Accuracy': cv_trainScores_acc,
                            'Validate Accuracy': cv_testScores_acc,
                            'Train Precision': cv_trainScores_prec,
                            'Validate Precision': cv_testScores_prec,
                            'Train Recall': cv_trainScores_recall,
                            'Validate Recall': cv_testScores_recall,
                            'Train F1': cv_trainScores_f1,
                            'Validate F1': cv_testScores_f1,
                            'Train ROC': cv_trainScores_roc,
                            'Validate ROC': cv_testScores_roc}, index = model_names).T
                            
    return cv_evalSummary


def hyperparameters_tuning(pipe, param_grid, X, y):
    
    # split the dataset
    shuffle_split = ShuffleSplit(test_size = 0.20, train_size = 0.80, n_splits = 5, random_state = 0)
    # perform hyperparameter tuning using gridsearchcv
    gs_res = GridSearchCV(pipe, param_grid = param_grid,
                          cv = shuffle_split, scoring = 'accuracy', verbose = 3)
    # fitted the model for grid search
    gs_res.fit(X, y)
    return gs_res


def confusionMatrix(title, y_true, y_predict):
    cf_matrix = confusion_matrix(y_true, y_predict)
    
    cf_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    cf_counts = ["{0:0.0f}".format(val) for val in cf_matrix.flatten()]
    cf_labels = np.asarray([f"{names}\n{counts}" for names, counts in zip(cf_names, cf_counts)]).reshape(2,2)

    
    ax = sns.heatmap(cf_matrix, fmt = "", annot = cf_labels, cmap = "viridis")
    ax.set_title("Confusion Matrix for " + title, fontweight = "bold", fontsize = 10)
    ax.xaxis.set_ticklabels(['No Hypertension', 'Hypertension'])
    ax.yaxis.set_ticklabels(['No Hypertension', 'Hypertension'])
    plt.show()


def rocCurve(false_pos, true_pos):
    plt.plot(false_pos, true_pos, color = "crimson", lw = 2.5)
    plt.plot([0, 1],[0, 1], color = "navy", lw = 2.5, ls = "--")
    plt.title("ROC Curve for Random Forest Classifier (Hypertension)", fontsize = 10, fontweight = "bold")
    plt.xlabel("False Positive Rate (1-Specificity)", fontsize = 10, fontweight = "bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize = 10, fontweight = "bold")
    plt.grid(color = 'black', linestyle = '--', linewidth = '0.8')
    #plt.legend("loc = lower right")
    plt.show()


def classificationReport(y_true, y_pred):
    classReport = classification_report(y_true, y_pred, output_dict = False, target_names = ['No Hypertension', 'Hypertension'])
    return classReport



def main():
    ################################### Data Extraction ######################################
    # call the function to load the datasets
    bp_df = load_datasets()
    
    # display the first five records from the datasets
    print(bp_df.head())
    
    ######################### Checking of Missing Data and LabelEncoding ##########################
    
    # call the function to generate a heatmap to identify the row missing data
    missing_data(bp_df)
    
    # call the function to convert the categorical text to num for class (Regular = 1 and Hypertension = 0)
    bp_df = categorical_class_to_num(bp_df)
    
    
    
    ############################# Data Exploration ###########################################
    # print the data info
    print(bp_df.info())
    
    # call the function to perform regression analysis to check the relationship between variables
    regression_analysis(bp_df)
    
    # call the function to generate the distribution curves to check the skewness and outliers data
    feature_names = ['bmi', 'dbp', 'sbp']
    distribution_curve(bp_df, feature_names)
    
    # call the function to generate the box-plot to check the outliers data
    boxplot(bp_df)
    
    # call the function to display the histogram of every variable in a dataframe
    histogram(bp_df)
    
    
    
    ##################### Removing Outliers using Isolation Forest ###################################
    # summarize the number of the dataset before removing outliers
    X_features = bp_df.drop("outcome", axis = 1)
    y_target = bp_df['outcome']
    
    iso = IsolationForest(contamination = 0.01, random_state = 1)
    iso_predict = iso.fit_predict(X_features)
    
    # to obtain the anomaly score 
    iso_scores = iso.decision_function(X_features)
    
    # display the anomaly score 
    sns.distplot(iso_scores, kde = False, bins = 50)
    plt.title("Distribution of the Anomaly Scores using Isolation Forest for Hypertension", fontweight = 'bold', fontsize = 14)
    plt.xlabel("Average Path Length", fontweight = 'bold', fontsize = 12)
    plt.ylabel("Count", fontweight = 'bold', fontsize = 12)
    plt.show()
    
    # identify the outliers using the selected threshold value
    outliers_index = np.where(iso_scores < -0.01)[0]
    
    # filter out the outliers from the original dataset based on the outliers index
    x_features_iso = X_features[~X_features.index.isin(outliers_index)]
    y_target_iso = y_target[~y_target.index.isin(outliers_index)]
    
    # convert the x_features_iso and y_target_iso into dataframe
    X_features = pd.DataFrame(data = x_features_iso, columns = X_features.columns)
    y_target = pd.Series(data = y_target_iso, name = 'outcome')
    
    
    
    ############################### Train - Test Split #########################################
    X_train, X_test, y_train, y_test = train_test_split(X_features, 
                                                        y_target, 
                                                        test_size = 0.30, 
                                                        random_state = 42, 
                                                        stratify = y_target)
    
    
    
    ############################# Resampling Data using SMOTE-Tomek ############################
    ####### Before Resampling ########
    # print the distribution count of the imbalanced class labels before resampling
    print("\nBefore Resampling for Hypertension using SMOTE-Tomek:")
    print(y_train.value_counts(ascending = True))
    print()
    
    
    # get the index from the class labels
    index = y_train.index
    # subset the index by a condition to get only the indices of rows which satisfy the condition. 
    # call tolist() on this result to get a list of these indices.
    cond_bp = y_train == 0
    cond_no_bp = y_train == 1
    bp = index[cond_bp].tolist()
    no_bp = index[cond_no_bp].tolist()
    
    
    ### display the distribution of the imbalanced class labels before resampling
    # set figure size
    plt.figure(figsize = (10, 6))
    # plot a countplot
    ax = sns.countplot(data = [bp, no_bp], palette = 'Set3')
    # set title
    ax.set_title("Distribution of Imbalanced Class Labels (Hypertension) Before Resampling", fontweight = 'bold', fontsize = 14)
    # set axis labels
    ax.set_xlabel("Outcome", fontweight = 'bold', fontsize = 14)
    ax.set_ylabel("No. of Outcome", fontweight = 'bold', fontsize = 14)
    # set axis ticklabels
    ax.set_xticklabels(['Hypertension (Class: 0)', 'No Hypertension (Class: 1)'], fontweight = 'bold', fontsize = 12)
    # remove the top and right spine of the plot
    sns.despine()
    # show the plot
    plt.show()
    
    
    ##### call the function to balanced the datasets
    X_smt, Y_smt = balanced_datasets(X_train, y_train)
    
    
    ####### After Resampling ########
    # print the distribution of the imbalanced class labels after resampling
    print("\nAfter Resampling for Hypertension using SMOTE-Tomek: ")
    print(Y_smt.value_counts(ascending = True))
    print()
    
    
    # get the index from the class labels
    index = Y_smt.index
    # subset the index by a condition to get only the indices of rows which satisfy the condition. 
    # call tolist() on this result to get a list of these indices.
    cond_bp_res = Y_smt == 0
    cond_no_bp_res = Y_smt == 1
    bp_res = index[cond_bp_res].tolist()
    no_bp_res = index[cond_no_bp_res].tolist()
    
    
    ##### display the distribution of the imbalanced class labels after resampling
    # set figure size
    plt.figure(figsize = (10, 6))
    # plot a countplot
    ax = sns.countplot(data = [bp_res, no_bp_res], palette = 'Set3')
    # set title
    ax.set_title("Distribution of Imbalanced Class Labels (Hypertension) After Resampling", fontweight = 'bold', fontsize = 14)
    # set axis labels
    ax.set_xlabel("Outcome", fontweight = 'bold', fontsize = 14)
    ax.set_ylabel("No. of Outcome", fontweight = 'bold', fontsize = 14)
    # set axis ticklabels
    ax.set_xticklabels(['Hypertension (Class: 0)', 'No Hypertension (Class: 1)'], fontweight = 'bold', fontsize = 12)
    # remove the top and right spine of the plot
    sns.despine()
    # show the plot
    plt.show()
    
    
    
    ######################## Model Selection based on Cross-Validation Evaluation ######################
    
    # initialize the ensemble classifier models with their default parameters and add them into a model list
    classifier_models = [('Random Forest', RandomForestClassifier(random_state = 42)),
                         ('Gradient Boosting', GradientBoostingClassifier(random_state = 42)),
                         ('Extra Trees', ExtraTreesClassifier(random_state = 42))]
    
    model_names = list()
    cv_trainScores_acc = list()
    cv_testScores_acc = list()
    cv_trainScores_prec = list()
    cv_testScores_prec = list()
    cv_trainScores_recall = list()
    cv_testScores_recall = list()
    cv_trainScores_f1 = list()
    cv_testScores_f1 = list()
    cv_trainScores_roc = list()
    cv_testScores_roc = list()
        
    for name, models in classifier_models:
        
        # define pipeline
        pipe = Pipeline(steps = [('scaler', MinMaxScaler()),('power', PowerTransformer()), ('model', models)])
        
        
        # call the function to perform cross validation evaluation 
        cvScores = cross_validation_eval(pipe, X_smt, Y_smt)
        
        
        # store all the respective scores from respective models into list
        cv_trainScores_acc.append(cvScores['train_accuracy'].mean())
        cv_testScores_acc.append(cvScores['test_accuracy'].mean())
        cv_trainScores_prec.append(cvScores['train_precision_macro'].mean())
        cv_testScores_prec.append(cvScores['test_precision_macro'].mean())
        cv_trainScores_recall.append(cvScores['train_recall_macro'].mean())
        cv_testScores_recall.append(cvScores['test_recall_macro'].mean())
        cv_trainScores_f1.append(cvScores['train_f1_macro'].mean())
        cv_testScores_f1.append(cvScores['test_f1_macro'].mean())
        cv_trainScores_roc.append(cvScores['train_roc_auc'].mean())
        cv_testScores_roc.append(cvScores['test_roc_auc'].mean())
        model_names.append(name)
    
    # call function to consolidate the cross-validation scores into dataframe
    baseline_models_CV_eval = cross_validation_summary(model_names, 
                                          cv_trainScores_acc, 
                                          cv_testScores_acc,
                                          cv_trainScores_prec,
                                          cv_testScores_prec,
                                          cv_trainScores_recall,
                                          cv_testScores_recall,
                                          cv_trainScores_f1,
                                          cv_testScores_f1,
                                          cv_trainScores_roc,
                                          cv_testScores_roc)
    
    # display the cross-validation evaluation 
    print("\nCross-Validation Scores for Hypertension: ")
    print(baseline_models_CV_eval)
    
    
    # print classification report for each model
    classifierModels = [RandomForestClassifier(random_state = 42),
                 GradientBoostingClassifier(random_state = 42),
                 ExtraTreesClassifier(random_state = 42)]
    
    for i in range(len(model_names)):
        
        pipe = Pipeline(steps = [('scaler', MinMaxScaler()), ('power', PowerTransformer()), ('model', classifierModels[i])])
        test_y_pred = cross_val_predict(pipe, X_smt, Y_smt, cv = 5)
        classificationReports = classificationReport(Y_smt, test_y_pred)
        print("Classification Report for {} (Hypertension)".format(model_names[i]))
        print(classificationReports)
        print()
    
    
    
    ###################### Hyperparameter Optimization on Random Forest Classifier ##########################
    #build pipeline
    pipe = Pipeline([('scaler', MinMaxScaler()), ('power', PowerTransformer()),("rfc", RandomForestClassifier(random_state = 42, 
                                                                                  max_leaf_nodes = None,
                                                                                  max_features = "sqrt"))])
    
    # set parameter grid
    param_grid = {'rfc__n_estimators': [50, 100, 150, 200, 250, 300, 350],
                  'rfc__criterion': ['gini', 'entropy'],
                  'rfc__max_depth': np.arange(5, 10),
                  'rfc__min_samples_leaf': np.arange(2, 7)}
    
    
    # call the function to perform hyperparameter tuning 
    gridResult = hyperparameters_tuning(pipe, param_grid, X_smt, Y_smt)
    
    # display the best parameters after tuning
    print("\nHyperparameter Optimization Process for Random Forest Classifier (Hypertension): ")
    print("=======================================================================================")
    print("The best score is {}".format(gridResult.best_score_))
    print("The best params is {}".format(gridResult.best_params_))
    print("The best estimator is {}".format(gridResult.best_estimator_))
    
    # store the best params into Series
    best_params_df = pd.Series(gridResult.best_params_)
    print(best_params_df)
    
    
    
    ######################## Rebuild Model (Random Forest) using Best Parameters ##################
    
    # From Hyperparameter Tuning: 
    # criterion = gini, max_depth = 5,  n_estimators = 50, min_samples_leaf = 2
    
    rfc_model = RandomForestClassifier(random_state = 42,
                                            max_leaf_nodes = None, 
                                            max_features = "sqrt",
                                            n_estimators = int(best_params_df.loc['rfc__n_estimators']),
                                            criterion = best_params_df.loc['rfc__criterion'],
                                            max_depth = int(best_params_df.loc['rfc__max_depth']),
                                            min_samples_leaf = int(best_params_df.loc['rfc__min_samples_leaf']))
    
    
    ####################### Evaluate the Rebuild Model - Random Forest ##########################
    pipe_rfc = Pipeline([('scaler', MinMaxScaler()),('power', PowerTransformer()), ('rfc', rfc_model)])
    rfc_cv_eval = pd.DataFrame(cross_validation_eval(pipe_rfc, X_smt, Y_smt))
    
    rfc_cv_results = pd.DataFrame({
                                   'Train Accuracy': rfc_cv_eval['train_accuracy'].mean(),
                                   'Validate Accuracy': rfc_cv_eval['test_accuracy'].mean(),
                                   'Train Precision': rfc_cv_eval['train_precision_macro'].mean(),
                                   'Validate Precision': rfc_cv_eval['test_precision_macro'].mean(),
                                   'Train Recall': rfc_cv_eval['train_recall_macro'].mean(),
                                   'Validate Recall': rfc_cv_eval['test_recall_macro'].mean(),
                                   'Train F1': rfc_cv_eval['train_f1_macro'].mean(),
                                   'Validate F1': rfc_cv_eval['test_f1_macro'].mean(),
                                   'Train ROC': rfc_cv_eval['train_roc_auc'].mean(),
                                   'Validate ROC': rfc_cv_eval['test_roc_auc'].mean()}, index = ['Random Forest']).T
    
    
    
    # display the cross-validation scores before and after hyperparameter optimization 
    print("\nCross-Validation Scores (Before Hyperparameter Optimization) for Random Forest Classifier (Hypertension): ")
    print(baseline_models_CV_eval['Extra Trees'])
    print("\nCross-Validation Scores (After Hyperparameter Optimization) for Random Forest Classifier (Hypertension): ")
    print(rfc_cv_results)
    
    # call the function to generate a confusion matrix
    rfc_y_test_predict = cross_val_predict(pipe_rfc, X_smt, Y_smt, cv = 5)
    confusionMatrix("Random Forest Classifier (Hypertension)", Y_smt, rfc_y_test_predict)
    
    # call the function to generate a ROC-AUC Curve
    rfc_y_test_proba = cross_val_predict(pipe_rfc, X_smt, Y_smt, cv = 5, method = "predict_proba")[:,-1]
    fpr, tpr, thresholds = roc_curve(Y_smt, rfc_y_test_proba, pos_label = 1)
    rocCurve(fpr, tpr)
    
    # display the classification report
    classification_Report = classificationReport(Y_smt, rfc_y_test_predict)
    print("\nClassification Report for Random Forest with Hyperparameter Tuning (Hypertension)")
    print(classification_Report)
    
    # fit the model on the training dataset
    pipe_rfc.fit(X_smt, Y_smt)
    
    
    
    ###################################### Save Model #########################################
    pickle.dump(pipe_rfc, open(r'C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\3. Source_Codes\hypertensionModel.pkl', 'wb'))
    
    
    ############################ Export Testing Set to CSV file ################################
    testing_set = pd.concat([X_test, y_test], axis = 1)
    testing_set.to_csv(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\test_set\bp_testing_set.csv", header = True, index = False)


def test_proposed_model():
    ########################### Evaluate the Model using Testing Set ###########################
    hypertension_model = pickle.load(open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\3. Source_Codes\hypertensionModel.pkl", "rb"))
    
    # retrieve the file from testing set
    testing = pd.read_csv(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\test_set\bp_testing_set.csv")
    
    X_test_features = testing.drop("outcome", axis = 1)
    y_test_class = testing["outcome"]
    
    # predict using testing set
    test_predict = hypertension_model.predict(X_test_features)
    test_predict_proba = hypertension_model.predict_proba(X_test_features)
    
    
    # evaluate the performance metrics
    print("Performance Metrics of the Random Forest with Hyperparameter Tuning (Hypertension) using Testing Set: ")
    print("=======================================================================================================")
    print("Accuracy Score: ", accuracy_score(y_test_class, test_predict))
    print("Precision Score: ", precision_score(y_test_class, test_predict, average = "macro"))
    print("Recall Score: ", recall_score(y_test_class, test_predict, average = "macro"))
    print("F1 Score: ", f1_score(y_test_class, test_predict, average = "macro"))
    print("ROC Score: ", roc_auc_score(y_test_class, test_predict))
    print()
    
    
    # call the function to generate the classification report
    classification_report_test = classificationReport(y_test_class, test_predict)
    print("Classification Report for Random Forest with Hyperparameter Tuning (Hypertension) using Testing Set: ")
    print(classification_report_test)
    
    # call the function to generate a confusion matrix
    confusionMatrix("Random Forest Classifier (Hypertension) from Testing Set", y_test_class, test_predict)

    # call the function to generate the roc auc score
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test_class, np.argmax(test_predict_proba, axis = 1), pos_label = 1)
    rocCurve(fpr_test, tpr_test)
    

# call the function to perform the whole series of ML pipeline 
main()

# call the function to test the proposed ML model
test_proposed_model()
