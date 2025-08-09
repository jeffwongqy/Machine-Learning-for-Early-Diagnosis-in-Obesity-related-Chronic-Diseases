# import relevant libraries
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTETomek
import pickle
from sklearn.ensemble import IsolationForest
from matplotlib import pyplot 



def load_datasets():
    conn = sqlite3.connect(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\3. Source_Codes\databases\chronic_diseases.sqlite3")
    obesity_df = pd.read_sql("SELECT gender, age, height, weight, family_history_with_overweight, \
                             caloric_food, vegetables, number_meals, food_between_meals, smoke, water, \
                             calories, activity, technology, alcohol, obesity_level FROM obesityLevel", conn)
    
    # create a set of copy of diabetes dataset
    copy_obesity_df = obesity_df.copy()
    
    # rename the target column
    copy_obesity_df.rename(columns = {'obesity_level': 'outcome', 'family_history_with_overweight': 'family_history_ow'}, inplace = True)
    
    # manipulate some of the target values
    copy_obesity_df['outcome'].replace([(3, 4), (5, 6, 7)], (3, 4), inplace = True)
    
    conn.close()
    return copy_obesity_df


def missing_data(obs_df):
    plt.title("Investigate the Missing Values in Obesity Dataset")
    sns.heatmap(obs_df.isnull(), annot = False)
    plt.show()

    
def categorical_text_to_num(obs_df, features):
   for feature in features:
       le = LabelEncoder()
       obs_df[feature] = le.fit_transform(obs_df[feature])
       
   return obs_df


def regression_analysis(obs_df):
    # generate a pairplot with regression plot to illustrate the relationship between variables
    sns.pairplot(obs_df, kind = "reg", hue = "outcome", palette = "coolwarm")
    plt.show()
    
    # generate a heatmap to illustrate the Pearson correlation between variables
    plt.figure(figsize = (15, 6))
    sns.heatmap(obs_df.corr(), annot = True, cmap = "Blues")
    plt.title("Heatmap of Obesity", fontweight = 'bold', fontsize = 14)
    plt.tight_layout()
    plt.show()

def distribution_curve(obs_df, feature_names):
    for feature in feature_names:
        sns.displot(x = feature, data = obs_df, hue = "outcome", kde = True, palette = "coolwarm")
        plt.xlabel(feature, fontsize = 14, fontweight = "bold")
        plt.ylabel("Count", fontsize = 14, fontweight = "bold")
        plt.show()
        print()

def boxplot(obs_df):
    sns.boxplot(data = obs_df.drop('outcome', axis = 1), orient = 'h', palette = 'Set2')
    plt.show()


def histogram(obs_df):
    obs_df.hist()
    plt.tight_layout()
    pyplot.show()

def balanced_datasets(X, y):
    oversample_SMT = SMOTETomek(random_state = 1)
    x_smt, y_smt = oversample_SMT.fit_resample(X, y)
    return x_smt, y_smt


def cross_validation_eval(model, X_smt, Y_smt):
    
    shuffle_split = ShuffleSplit(test_size = 0.20, train_size = 0.80, n_splits = 5, random_state = 0)
    
    # compute the accuracy, precision, recall and f1-scores
    cv_result = cross_validate(model, 
                               X_smt, 
                               Y_smt, 
                               cv = shuffle_split,
                               n_jobs = 1,
                               scoring = ('accuracy', 'precision_macro', 'recall_macro', 'f1_macro'),
                               return_train_score = True)
    
    # compute the ROC scores
    myscore = make_scorer(roc_auc_score, multi_class='ovo',needs_proba=True)
    roc_scores = cross_validate(model,
                                X_smt,
                                Y_smt,
                                cv = shuffle_split,
                                n_jobs = 1,
                                scoring = myscore,
                                return_train_score = True)
    return cv_result, roc_scores


    

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
    plt.figure(figsize = (10, 6))
    cf_matrix = confusion_matrix(y_true, y_predict)
    
    ax = sns.heatmap(cf_matrix, fmt = "", annot = True, cmap = "viridis")
    ax.set_title("Confusion Matrix for " + title, fontweight = "bold", fontsize = 10)
    ax.xaxis.set_ticklabels(['Obesity', 'Overweight', 'Normal Weight', 'Underweight'])
    ax.yaxis.set_ticklabels(['Obesity', 'Overweight', 'Normal Weight', 'Underweight'])
    plt.tight_layout()
    plt.show()
    

def rocCurve(false_pos, true_pos):
    plt.plot(false_pos, true_pos, color = "crimson", lw = 2.5)
    plt.plot([0, 1],[0, 1], color = "navy", lw = 2.5, ls = "--")
    plt.title("ROC Curve for Random Forest Classifier (Obesity)", fontsize = 10, fontweight = "bold")
    plt.xlabel("False Positive Rate (1-Specificity)", fontsize = 10, fontweight = "bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize = 10, fontweight = "bold")
    plt.grid(color = 'black', linestyle = '--', linewidth = '0.8')
    #plt.legend("loc = lower right")
    plt.show()  
    
def classificationReport(y_true, y_pred):
    classReport = classification_report(y_true, y_pred, output_dict = False, target_names = ['Underweight', 'Normal Weight', 'Overweight', 'Obesity'])
    return classReport



def main():
    ################################### Data Extraction ######################################
    ##### Data Extraction
    # call the function to load the datasets
    obs_df = load_datasets()
    
    # display the first five records from the datasets
    print(obs_df.head())
    
    
    ######################### Checking of Missing Data and Encoding ##########################
    # call the function to generate a heatmap to identify the row missing data
    missing_data(obs_df)
    
    # call the function to convert the categorical text to num
    features = ['gender', 'family_history_ow', 'caloric_food', 'smoke', 'calories']
    obese_df = categorical_text_to_num(obs_df, features)
    
    
    
    ############################# Data Exploration ###########################################
    # print data info
    print(obese_df.info())
    
    # call the function to perform regression analysis to check the relationship between variables
    regression_analysis(obs_df)
    
    # call the function to generate the distribution curves to check the skewness and outliers data
    feature_names = ['age', 'height', 'weight']
    distribution_curve(obs_df, feature_names)
    
    # call the function to generate the box-plot to check the outliers data
    boxplot(obs_df)
    
    # call the function to display the histogram of every variable in a dataframe
    histogram(obs_df)
    
    
    
    ##################### Removing Outliers using Isolation Forest #############################
    # summarize the number of the dataset before removing outliers
    X_features = obs_df.drop("outcome", axis = 1)
    y_target = obs_df['outcome']
    
    # define and identify outliers in the dataset
    iso = IsolationForest(contamination = 0.1, random_state = 1)
    iso_pred = iso.fit_predict(X_features)
    
    # to obtain the anomaly score 
    iso_scores = iso.decision_function(X_features)
    
    # display the anomaly score 
    sns.distplot(iso_scores, kde= False, color = 'crimson', bins = 100)
    plt.title("Distribution of the Anomaly Scores using Isolation Forest for Obesity", fontweight = 'bold', fontsize = 14)
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
    
    after_rOutlier = pd.concat([X_features, y_target], axis = 1)
    boxplot(after_rOutlier)
    
    
    ################################## Train-Test Split #######################################
    X_train, X_test, y_train, y_test = train_test_split(X_features, 
                                                        y_target, 
                                                        test_size = 0.30, 
                                                        random_state = 42, 
                                                        stratify = y_target)
    
    
    
    ############################# Resampling Data using SMOTE-Tomek ############################
    ####### Before Resampling ########
    # print the distribution count of the imbalanced class labels before resampling
    print("\nBefore Resampling for Obesity Level using SMOTE-Tomek:")
    counter = Counter(y_train)
    for cls, n in sorted(counter.items()):
        percent = n/len(y_train) * 100
        print('Class = {}, n = {} ({:.3f}%)'.format(cls, n, percent))
    
    
    # plot the distribution of the imbalanced class labels before resampling
    # set figure size
    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_axes([0,0,1,1])
    # plot a counterplot
    obs_cat = ['Underweight Level (Class: 1)', 'Normal Weight Level (Class: 2)', 'Overweight Level (Class: 3)', 'Obesity Level (Class: 4)']
    ax.bar(obs_cat, sorted(counter.values()), color = ['r', 'g', 'b', 'm'])
    # set title
    ax.set_title("Distribution of Imbalanced Class Labels (Obesity Level) Before Resampling", fontweight = 'bold', fontsize = 14)
    # set axis labels
    ax.set_xlabel("Outcome ", fontweight = 'bold', fontsize = 14)
    ax.set_ylabel("No. of Outcome", fontweight = 'bold', fontsize = 14)
    # remove the top and right spine of the plot
    sns.despine()
    plt.show()
    
    
    
    ##### call the function to balanced the datasets
    X_smt, Y_smt = balanced_datasets(X_train, y_train)
    
    
    
    ####### After Resampling ########
    # print the distribution of the imbalanced class labels after resampling
    print("\nAfter Resampling for Obesity Level using SMOTE-Tomek:")
    counter_res = Counter(Y_smt)
    for cls, n in sorted(counter_res.items()):
        percent_res = n/len(Y_smt) * 100
        print('Class = {}, n = {} ({:.3f}%)'.format(cls, n, percent_res))
    
    
    # plot the distribution of the imbalanced class labels before resampling
    # set figure size
    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_axes([0,0,1,1])
    # plot a counterplot
    obs_cat = ['Underweight Level (Class: 1)', 'Normal Weight Level (Class: 2)', 'Overweight Level (Class: 3)', 'Obesity Level (Class: 4)']
    ax.bar(obs_cat, sorted(counter_res.values()), color = ['r', 'g', 'b', 'm'])
    # set title
    ax.set_title("Distribution of Imbalanced Class Labels (Obesity Level) After Resampling", fontweight = 'bold', fontsize = 14)
    # set axis labels
    ax.set_xlabel("Outcome ", fontweight = 'bold', fontsize = 14)
    ax.set_ylabel("No. of Outcome", fontweight = 'bold', fontsize = 14)
    # remove the top and right spine of the plot
    sns.despine()
    plt.show()
    
    
    
    
    ########################## Model Selection based on Cross-Validation ######################
    
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
        
        # build pipeline
        pipe = Pipeline(steps = [('scaler', StandardScaler()), ('model', models)])
        
        # call function to compute cross validation evaluation
        cvScores, rocScores = cross_validation_eval(pipe, X_smt, Y_smt)
        
        # store all the respective scores from respective models into list
        cv_trainScores_acc.append(cvScores['train_accuracy'].mean())
        cv_testScores_acc.append(cvScores['test_accuracy'].mean())
        cv_trainScores_prec.append(cvScores['train_precision_macro'].mean())
        cv_testScores_prec.append(cvScores['test_precision_macro'].mean())
        cv_trainScores_recall.append(cvScores['train_recall_macro'].mean())
        cv_testScores_recall.append(cvScores['test_recall_macro'].mean())
        cv_trainScores_f1.append(cvScores['train_f1_macro'].mean())
        cv_testScores_f1.append(cvScores['test_f1_macro'].mean())
        cv_trainScores_roc.append(rocScores['train_score'].mean())
        cv_testScores_roc.append(rocScores['test_score'].mean())
        model_names.append(name)
    
    # call the function to consolidate the cross-validation scores into dataframe
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
    print("\nCross-Validation Scores for Obesity: ")
    print(baseline_models_CV_eval)
    
    
    # print classification report for each model
    classifierModels = [RandomForestClassifier(random_state = 42),
                 GradientBoostingClassifier(random_state = 42),
                 ExtraTreesClassifier(random_state = 42)]
    
    for i in range(len(model_names)):
        pipe = Pipeline(steps = [('scaler', StandardScaler()), ('model', classifierModels[i])])
        test_y_pred = cross_val_predict(pipe, X_smt, Y_smt, cv = 5)
        classificationReports = classificationReport(Y_smt, test_y_pred)
        print("Classification Report for {}".format(model_names[i]))
        print(classificationReports)
        print()
        
    
    
    ################ Hyperparameter Optimization on Random Forest #########################
    # build pipeline
    pipe = Pipeline([('scaler', StandardScaler()),("rfc", RandomForestClassifier(random_state = 42, 
                                                  max_leaf_nodes = None,
                                                  max_features = 'sqrt'))])
    
    # set parameter grid
    param_grid = {'rfc__n_estimators': [80, 100, 150, 200, 250, 300],
                  'rfc__max_depth': np.arange(2, 7),
                  'rfc__criterion': ['gini', 'entropy'],
                  'rfc__min_samples_split': np.arange(2, 5)}
    
    # call the function to perform hyperparameter tuning
    gridResult = hyperparameters_tuning(pipe, param_grid, X_smt, Y_smt)
    
    # display the best parameters after tuning
    print("\nHyperparameter Optimization Process for Random Forest Classifer (Obesity): ")
    print("=======================================================================================")
    print("The best score is {}".format(gridResult.best_score_))
    print("The best params is {}".format(gridResult.best_params_))
    print("The best estimator is {}".format(gridResult.best_estimator_))
    
    # store the best params into Series
    best_params_df = pd.Series(gridResult.best_params_)
    print(best_params_df)
    
    
    
    ######################## Rebuild Model (Random Forest) using Best Parameters ##################
    
    # From Hyperparameter Tuning: 
    # criterion = entropy, max_depth = 6,  n_estimators = 80, min_samples_split = 3
    
    rfc_model = RandomForestClassifier(random_state = 42,
                                            max_features = 'sqrt',
                                            max_leaf_nodes = None, 
                                            n_estimators = int(best_params_df.loc['rfc__n_estimators']),
                                            max_depth = int(best_params_df.loc['rfc__max_depth']),
                                            criterion = best_params_df.loc['rfc__criterion'],
                                            min_samples_split = int(best_params_df.loc['rfc__min_samples_split']))
    
    
    ####################### Evaluate the Rebuild Model - Random Forest ##########################
    pipe_rfc = Pipeline([('scaler', StandardScaler()), ('rfc', rfc_model)])
    rfc_cv_eval, rfc_rocScores = cross_validation_eval(pipe_rfc, X_smt, Y_smt)
    
    
    rfc_cv_results = pd.DataFrame({
                                   'Train Accuracy': rfc_cv_eval['train_accuracy'].mean(),
                                   'Validate Accuracy': rfc_cv_eval['test_accuracy'].mean(),
                                   'Train Precision': rfc_cv_eval['train_precision_macro'].mean(),
                                   'Validate Precision': rfc_cv_eval['test_precision_macro'].mean(),
                                   'Train Recall': rfc_cv_eval['train_recall_macro'].mean(),
                                   'Validate Recall': rfc_cv_eval['test_recall_macro'].mean(),
                                   'Train F1': rfc_cv_eval['train_f1_macro'].mean(),
                                   'Validate F1': rfc_cv_eval['test_f1_macro'].mean(),
                                   'Train ROC': rfc_rocScores['train_score'].mean(),
                                   'Validate ROC': rfc_rocScores['test_score'].mean()}, index = ['Random Forest']).T
    
    
    # display the cross-validation scores before and after hyperparameter optimization 
    print("\nCross-Validation Scores (Before Hyperparameter Optimization) for Random Forest Classifier (Obesity): ")
    print(baseline_models_CV_eval['Random Forest'])
    print("\nCross-Validation Scores (After Hyperparameter Optimization) for Random Forest Classifier (Obesity): ")
    print(rfc_cv_results)
    
    
    # call the function to generate a confusion matrix
    rfc_y_test_predict = cross_val_predict(pipe_rfc, X_smt, Y_smt, cv = 5)
    confusionMatrix("Random Forest Classifier (Obesity Level)", Y_smt, rfc_y_test_predict)
    
    
    # call the function to generate a ROC-AUC Curve
    gbc_y_test_proba = cross_val_predict(pipe_rfc, X_smt, Y_smt, cv = 5, method = "predict_proba")[:,-1]
    fpr, tpr, thresholds = roc_curve(Y_smt, gbc_y_test_proba, pos_label = 4)
    rocCurve(fpr, tpr)
    
    # display the classification report
    print()
    print("\nClassification Report for Random Forest with Hyperparameter Tuning (Obesity)")
    classification_Report = classificationReport(Y_smt, rfc_y_test_predict)
    print(classification_Report)
    
    # fit the model on the training dataset
    pipe_rfc.fit(X_smt, Y_smt)
    
    
    ###################################### Save Model #########################################
    pickle.dump(pipe_rfc, open(r'C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\3. Source_Codes\obesityModel.pkl', 'wb'))
    
    
    ############################ Export Testing Set to CSV file ################################
    testing_set = pd.concat([X_test, y_test], axis = 1)
    testing_set.to_csv(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\test_set\obesity_testing_set.csv", header = True, index = False)

def test_proposed_model():
    ########################### Evaluate the Model using Testing Set ###########################
    # predict using the loaded model 
    obesity_model = pickle.load(open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\3. Source_Codes\obesityModel.pkl", "rb"))
    
    # retrieve the file from testing set
    testing = pd.read_csv(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\test_set\obesity_testing_set.csv")
    
    X_test_features = testing.drop("outcome", axis = 1)
    y_test_class = testing["outcome"]
    
    # predict using testing set
    test_predict = obesity_model.predict(X_test_features)
    test_predict_proba = obesity_model.predict_proba(X_test_features)
    
    # evaluate the performance metrics
    print("Performance Metrics of the Random Forest with Hyperparameter Tuning (Obesity) using Testing Set: ")
    print("=======================================================================================================")
    print("Accuracy Score: ", accuracy_score(y_test_class, test_predict))
    print("Precision Score: ", precision_score(y_test_class, test_predict, average = "macro"))
    print("Recall Score: ", recall_score(y_test_class, test_predict, average = "macro"))
    print("F1 Score: ", f1_score(y_test_class, test_predict, average = "macro"))
    print("ROC Score: ", roc_auc_score(y_test_class, test_predict_proba, multi_class = 'ovo'))
    print()
    
    # call the function to generate the classification report
    classification_report_test = classificationReport(y_test_class, test_predict)
    print("Classification Report for Random Forest with Hyperparameter Tuning (Obesity) using Testing Set: ")
    print(classification_report_test)
    
    # call the function to generate a confusion matrix
    confusionMatrix("Random Forest Classifier (Obesity) from Testing Set", y_test_class, test_predict)
    
    # generate the ROC score
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test_class, np.argmax(test_predict_proba, axis = 1), pos_label = 4)
    rocCurve(fpr_test, tpr_test)


# call the function to perform the whole series of ML pipeline 
main()

# call the function to test the proposed ML model
test_proposed_model()
