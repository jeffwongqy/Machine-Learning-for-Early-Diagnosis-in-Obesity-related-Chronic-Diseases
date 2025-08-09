import pickle
import numpy as np
import streamlit as st
from PIL import Image
import sqlite3
from sqlite3 import Error

# load the respective machine learning models
diab_model = pickle.load(open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\3. Source_Codes\diabetesModel.pkl", "rb"))
bp_model = pickle.load(open(r'C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\3. Source_Codes\hypertensionModel.pkl', 'rb'))
obs_model = pickle.load(open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\3. Source_Codes\obesityModel.pkl", "rb"))

# set location of the database file
DBFILENAME = r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\3. Source_Codes\databases\usersPredictedOutcomes.db"


# define function for create SQL connection
def create_connection(db_filename):
    try:
        conn = sqlite3.connect(db_filename)
        return conn
    except Error as e:
        print(e)
    return None


# define function for obesity prediction
def obesity_predict(gender, age, height, weight, family_history_obesity, caloric_food, fruit_vegetables, num_of_meals, between_meals, smoking, water, calories, activity, technology, alcohol):
    
    # convert the gender into numerical type
    if gender == 'Female':
        gender = 0
    else:
        gender = 1
    
    # convert the family_history_obesity into numerical type
    if family_history_obesity == 'Yes':
        family_history_obesity = 1
    else:
        family_history_obesity = 0
    
    # convert the caloric food into numerical type
    if caloric_food == 'Yes':
        caloric_food = 1
    else:
        caloric_food = 0
    
    # convert the vegetables into numerical type
    if fruit_vegetables == 'I do not eat fruits and vegetables.':
        fruit_vegetables = 1
    elif fruit_vegetables == 'Frequently':
        fruit_vegetables = 2
    else:
        fruit_vegetables = 3
    
    # convert between meals into numerical type
    if between_meals == 'I do not consume foods between meals.':
        between_meals = 1
    elif between_meals == 'Sometimes':
        between_meals = 2
    elif between_meals == 'Frequently':
        between_meals = 3
    else:
        between_meals = 4
    
    # convert the smoking into numerical type
    if smoking == 'Yes':
        smoking = 1
    else:
        smoking = 0
        
    # convert water into numerical type
    if water == 'I do not always drink water.': 
        water = 1
    elif water == 'Sometimes':
        water = 2
    else:
        water = 3

    # convert calories into numerical type
    if calories == 'Yes':
        calories = 1
    else:
        calories = 0
    
    # convert activity into numerical type
    if activity == 'I do not exercise.':
        activity = 0
    elif activity == 'Sometimes':
        activity = 1
    elif activity == 'Frequently':
        activity = 2
    elif activity == 'Always':
        activity = 3
    
    # convert technology into numerical type
    if technology == 'Sometimes':
        technology = 0
    elif technology == 'Frequently':
        technology = 1
    else:
        technology = 2
    
    # convert alcohol into numerical type
    if alcohol == 'I do not drink alcohol.':
        alcohol = 1
    elif alcohol == 'Sometimes':
        alcohol = 2
    elif alcohol == 'Frequently':
        alcohol = 3
    elif alcohol == 'Always':
        alcohol = 4
    
    # convert the number_meals into integer type
    num_of_meals = int(num_of_meals)
    
    # combined the input data into an array list
    obesity_input_data = np.array([gender, age, height, weight, family_history_obesity, caloric_food, fruit_vegetables, num_of_meals, between_meals, smoking, water, calories, activity, technology, alcohol]).reshape(1, -1)
        
    # predict the obesity outcome
    obesityOutcome = obs_model.predict(obesity_input_data)
    obesityOutcomeProba = obs_model.predict_proba(obesity_input_data)
    
    return obesityOutcome, obesityOutcomeProba
    

# define function for diabetes prediction
def diab_predict(gender, glucose, bmi):
    
    # convert the gender into numerical type
    if gender == 'Female':
        gender = 0
    else:
        gender = 1
        
    # combined the input data into an array list
    diab_input_data = np.array([gender, glucose, bmi]).reshape(1, -1)
    
    # predict the diabetes outcome
    diabOutcome = diab_model.predict(diab_input_data)
    diabOutcomeProba = diab_model.predict_proba(diab_input_data)
        
    return diabOutcome, diabOutcomeProba


# define function for hypertension prediction
def hypertension_predict(bmi, sbp, dbp):
    
    # combined the input data into an array list
    bp_input_data = np.array([bmi, sbp, dbp]).reshape(1, -1)
    
    # predict the hypertension outcome
    bpOutcome = bp_model.predict(bp_input_data)
    bpOutcomeProba = bp_model.predict_proba(bp_input_data)
    
    return bpOutcome, bpOutcomeProba
    


# define function to display the obesity prediction outcome
def display_obesity_outcome(name, obesity_pred_class, obesity_outcome_proba):
    # display the outcome
    st.subheader("Outcomes: ")
    if obesity_pred_class == 1:
        obesity_pred_outcome = "At low risk for obesity-related diseases, but at risk of nutritional deficiency and osteoporosis. "
        obesity_outcome_prob = obesity_outcome_proba[:, 0]*100
        st.warning("Hi {}! You have about {}% chance at low risk for obesity-related diseases.".format(name, obesity_outcome_prob ))
        st.subheader("Recommendation: ")
        st.info("However, you are at risk of nutritional deficiency and osteoporosis. You are encouraged to eat a balanced meal and to seek medical advice if necessary. ")
    elif obesity_pred_class == 2:
        obesity_pred_outcome = "At low risk for obesity-related diseases."
        obesity_outcome_prob = obesity_outcome_proba[:, 1]*100
        st.success("Hi {}! You have about {}% chance at low risk for obesity-related diseases.".format(name, obesity_outcome_prob))
        st.subheader("Recommendation: ")
        st.info("Please continue to achieve a healthy weight by balancing your caloric input and physical activity. ")
    elif obesity_pred_class == 3:
        obesity_pred_outcome = "At moderate risk for obesity-related diseases."
        obesity_outcome_prob = obesity_outcome_proba[:, 2]*100
        st.warning("Hi {}! You have about {}% chance at moderate risk for obesity-related diseases.".format(name, obesity_outcome_prob))
        st.subheader("Recommendation: ")
        st.info("Please try to aim to lose at least 5% to 10% of your body weight over 6 to 12 months by increasing your physical activity and reducing caloric intake. You are encourage to use this app to check for possible risks of developing diabetes and hypertension. ")
    elif obesity_pred_class == 4:
        obesity_pred_outcome = "At high risk for obesity-related diseases."
        obesity_outcome_prob = obesity_outcome_proba[:, 3]*100
        st.error("Hi {}! You have about {}% chance at high risk for obesity-related diseases. ".format(name, obesity_outcome_prob))
        st.subheader("Recommendation:")
        st.info("Please try to aim to loss at least 5% to 10% of your body weight over 6 to 12 months by increasing your physical activity and reducing caloric intake. You should go for regular health screening to keep co-morbid conditions in check or use this app to check for possible risks of developing diabetes and hypertension.")
    return obesity_pred_outcome, obesity_outcome_prob


# define function to display the diabetes prediction outcome
def display_diabetes_outcome(name, diab_pred_class, diab_outcome_proba):
    # display the outcome
    st.subheader("Outcome:")
    if diab_pred_class == 0:
        diab_pred_outcome = "No Diabetes"
        diab_outcome_prob = diab_outcome_proba[:, 0]*100
        st.success("Hi {}! You have a {}% chance at low risk of developing diabetes.".format(name, diab_outcome_prob))
        st.subheader("Recommendation: ")
        st.info("Please continue to achieve a healthy weight by balancing your caloric input and physical activity. ")
    elif diab_pred_class == 1:
        diab_pred_outcome = "Pre-Diabetes"
        diab_outcome_prob = diab_outcome_proba[:, 1]*100
        st.warning("Hi {}! You have a {}% chance at moderate risk of developing pre-diabetes.".format(name, diab_outcome_prob))
        st.subheader("Recommendation: ")
        st.info("Please try to achieve a healthy weight by balancing your caloric input and physical activity over the next 6 to 12 months. ")
    elif diab_pred_class == 2:
        diab_pred_outcome = "Diabetes"
        diab_outcome_prob = diab_outcome_proba[:, 2]*100
        st.error("Hi {}! You have a {}% chance at high risk of developing diabetes.".format(name, diab_outcome_prob))
        st.subheader("Recommendation: ")
        st.info("Please try to achieve a healthy weight by balancing your caloric input and physical activity over the next 6 to 12 months. You should also consult your doctor as soon as possible. ")
    return diab_pred_outcome, diab_outcome_prob


# define function to display the hypertension prediction outcome
def display_hypertension_outcome(name, bp_pred_class, bp_outcome_proba):
     # display the outcome
    st.subheader("Outcome:")
    if bp_pred_class == 0:
        bp_pred_outcome = "Hypertension"
        bp_outcome_prob = bp_outcome_proba[:, 0]*100
        st.error("Hi {}! You have a {}% chance at high risk of developing hypertension.".format(name, bp_outcome_prob))
        st.subheader("Recommendation: ")
        st.info("Please try to achieve a healthy weight by balancing your caloric input and physical acitivity over the next 6 to 12 months. You should also consult your doctor as soon as possible. ")
    elif bp_pred_class == 1:
        bp_pred_outcome = "No Hypertension"
        bp_outcome_prob = bp_outcome_proba[:,-1]*100
        st.success("Hi {}! You have a {}% chance at low risk of developing hypertension".format(name, bp_outcome_prob))
        st.subheader("Recommendation: ")
        st.info("Please continue to achieve a healthy weight by balancing your caloric input and physical activity.")
    return bp_pred_outcome, bp_outcome_prob
    
    

# define function to insert the user info and prediction outcomes into obesity database
def insert_obesity_pred_outcome_sql(DBFILENAME, users_info):
    try:
        conn = create_connection(DBFILENAME)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO obesity_pred_outcome (name, 
                                                           identification_no, 
                                                           email_address,
                                                           gender,
                                                           age,
                                                           height,
                                                           weight,
                                                           family_history_obesity,
                                                           having_caloric_food,
                                                           having_fruit_vegetables,
                                                           num_of_meals,
                                                           between_meals,
                                                           smoking,
                                                           drinking_water, 
                                                           monitor_calories,
                                                           activity,
                                                           usage_of_technology,
                                                           drinking_alcohol,
                                                           predicted_class,
                                                           predicted_proba,
                                                           predicted_outcome) VALUES(?,?, ?, ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', users_info)
        conn.commit()
        conn.close()
    except Error as e:
        print(e)


# define function to insert the user info and prediction outcomes into diabetes database
def insert_diabetes_pred_outcome_sql(DBFILENAME, users_info):
    try:
        conn = create_connection(DBFILENAME)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO diab_pred_outcome (name, 
                                                          identification_no,
                                                          email_address,
                                                          gender,
                                                          age,
                                                          bmi, 
                                                          glucose, 
                                                          predicted_class,
                                                          predicted_proba,
                                                          predicted_outcome) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', users_info)
        conn.commit()
        conn.close()
    
    except Error as e:
        print(e)

# define function to insert the user info and prediction outcomes into hypertension database
def insert_hypertension_pred_outcome_sql(DBFILENAME, users_info):
    try:
        conn = create_connection(DBFILENAME)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO bp_pred_outcome(name,
                                                      identification_no,
                                                      email_address,
                                                      gender,
                                                      age, 
                                                      bmi, 
                                                      sbp,
                                                      dbp,
                                                      predicted_class,
                                                      predicted_proba,
                                                      predicted_outcome) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', users_info)
        conn.commit()
        conn.close()
    except Error as e:
        print(e)


def bmi_calculation():
    st.sidebar.header("BMI Calculator")
    st.sidebar.write("Calculate your Body Mass Index (BMI) here")
    height = st.sidebar.number_input(label = "Enter your height (in m): ", min_value = 0.01, format = "%.2f")
    weight = st.sidebar.number_input(label = "Enter your weight (in kg): ", min_value = 0.01, format = "%.2f")
    # compute the bmi 
    bmi = weight/(height * height)
    return bmi




def main():
    # create a navigation sidebar 
    nav_box = ['Home','Obesity', 'Diabetes Mellitus', 'Hypertension']
    nav = st.sidebar.selectbox("Please choose one to explore:", nav_box)


    # main page of the app
    if nav == 'Home':
        # add title to the main page
        st.title("Early Screening of Common Obesity-related Chronic Diseases")
        obesity_image = Image.open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\images\health_screening_img.jpg")
        st.image(obesity_image, output_format = "auto", width = 600)
        st.title("Some basic facts about this app...")
        st.write("__Q1: What is so unique about this app?__ ")
        st.write("""
                 - This app uses the supervised machine learning algorithms with the best hyperparameters to predict the respective user's obesity, diabetes, and hypertension status as follows: 
                     - __Obesity__: Random Forest Classifier
                     - __Diabetes__: Gradient Boosting Classifier
                     - __Hypertension__: Random Forest Classifier
                 """)
                 
        st.write("__Q2: What is the objective of developing this app?__ ")
        st.write(""" 
                 - Diabetes mellitus and hypertension are the two common obesity-related chronic diseases and millions of people from all over the world fall victim to them. 
                 We believe that there are some kinds of apps that are currenttly available that can keep track of calories, sugar level, lifestyle, blood glucose, weight management of individuals, and 
                 providing some suggestions about the foods and kinds of activities to manage and prevent both diabetes and hypertension. However, there is no application 
                 that has been found to investigate the risk of being a diabetic or hypertensive patient. Therefore, the objective of this project is to develop an application that based on
                 machine learning models to assess the likelihood of an individual developing both diabetes and hypertension without much assistance from any healthcare professional medical provider. 
                 """)
                 
        st.write("__Q3: Who are the target users using this app?__ ")
        st.write("""
                 - Any users who are self-conscious about own health status 
                 - Any users who seldom or have no intention to visit the clinics or hospitals on regular basis for health screening
                 
                 """)
                 
        st.write("__Q4: How does a general user can benefit from using this app?__ ")
        st.write("""
                 - This app would provide an individual to know their likelihood of being developing either diabetes or hypertension or both, and hence, reduce the time to visit the clinic office in person. 
                 - Being cost-effective as the users are able to know the diagnosis right on the spot and also provide the users the time to prevent and manage obesity-related chronic diseases by making them aware of their present condition. 
                 
                 """)
                 
        st.write("__Q5: What kind of early screenings does this app offer?__")
        st.write(""" 
                 - Based on the current global health situation, the common obesity-related chronic diseases such as diabetes and hypertension are the major health of concern.
                   At the moment, this app only provide obesity, diabetes, and hypertension prediction. ___You may wish to start your health screenings by exploring the drop-down menu at the sidebar on the top left-hand___.
                 
                 """)
        
    
    # obesity page
    if nav == 'Obesity':
        # display unique information about obesity detection 
        st.sidebar.header("What's so unqiue about this obesity prediction app? ")
        st.sidebar.write("This application uses __Random Forest Algorithm__ (Supervised ML) with the best hyperparameters to predict the obesity level in an individual. \
                            During the testing phase, the outcome shows that the random forest algorithm can be achievable  ___with at least 90% accuracy, precision, recall, and AUC.___ ")
        
        st.sidebar.write("_Evaluation outcome from the testing set_: ")
        obesity_rfc_outcome_img = Image.open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\images\obesity_rfc_outcome.jpg")
        st.sidebar.image(obesity_rfc_outcome_img, output_format = "auto", width = 600)
        
        st.sidebar.write("__NOTE__: The objective of the testing set is to evaluate the performance of the trained model\
                            and was unseen during the training phase. ")
        st.sidebar.markdown("___")
        
        
        # display information on obesity 
        st.sidebar.header("What's Obesity? ")
        obesity_sidebar_img = Image.open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\images\obesity_sidebar_img.jpg")
        st.sidebar.image(obesity_sidebar_img, output_format = "auto", width = 300)
        st.sidebar.write("__Obesity__ can be defined as the weight that is higher than what is considered healthy for a given weight. BMI is a main screening tool for obesity.")
        st.sidebar.write("Obesity is a complex health issue resulting from a combination of causes and individual factors such as genetics and behaviour.\
                            Behaviours can include physical activity, inactivity, dietary patterns, and other exposures. ")
        st.sidebar.write("Obesity is serious because it is often associated with poor mental health outcomes and reduced quality of life. \
                            Furthermore, obesity is also associated with the leading causes of death, including diabetes and high blood pressure. ")
        st.sidebar.write("You may click the link below to know more about obesity.(https://www.cdc.gov/obesity/adult/causes.html)")
        st.sidebar.markdown("___")
        
        # display information about obesity management and prevention
        st.sidebar.header("How can we manage and prevent the chances of developing obesity? ")
        obesity_sidebar_img2 = Image.open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\images\obesity_sidebar_img2.jpg")
        st.sidebar.image(obesity_sidebar_img2, output_format = "auto", width = 300)
        st.sidebar.write( """
                             - Have a balanced and calorie-controlled diet as recommended by GP or weight loss management health professional
                             - Be more physically active
                             - Eat healthy plant foods
                             - Exercise regularly for 150 to 300 minutes
                             - Less smoking
                            """)
        st.sidebar.write("You may click the link below to know more about management and prevention of obesity.(https://www.nhsinform.scot/illnesses-and-conditions/nutritional/obesity)")
        st.sidebar.markdown("___")
        
        
        # add title to the obesity page
        st.title("Obesity Prediction")
        
        # add header images on obesity page
        obesity_image = Image.open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\images\obesity.jpg")
        st.image(obesity_image, output_format = 'auto', width = 600)
        
        st.info("__Medical Disclaimer__: This platform is not serve as an alternative to medical advice from medical professional healthcare provider.\
                 If you have any specific questions about any medical matter, you should consult your doctor or other medical professional healthcare provider. ")
        st.write("__General guideline(s) to the user:__ ")
        st.write("1) You are ___required___ to fill up all the information in this form in less than 5 mins. ")
        
        # prompt the user for input
        with st.form('Form 1'):
            name = st.text_input(label = "Enter your name:")
            identification_no = st.text_input(label = "Enter your identification no.:")
            email = st.text_input(label = "Enter your personal email address:")
            gender = st.radio('Gender:*', ['Male', 'Female'])
            age = st.number_input(label = "Enter your age:*", min_value = 0, max_value = 100, format = "%d")
            height = st.number_input(label = "Enter your height (in m):* ", min_value = 0.00, format = "%.2f")
            weight = st.number_input(label = "Enter your weight (in kg):* ", min_value = 0.00, format = "%.2f")
            family_history_obesity = st.radio('Does your family have a history of obesity?* ', ['Yes', 'No'])
            caloric_food = st.radio("Do you often consume high caloric foods?* ", ['Yes', 'No'])
            fruit_vegetables =  st.radio("How often do you consume fruits and vegetables?* ", ['I do not eat fruits and vegetables.', 'Frequently', 'Often'])
            num_of_meals = st.radio("What is the number of main meals per day?* ", [1, 2, 3, 4])
            between_meals = st.radio("How often do you consume foods between meals?* ", ['I do not consume foods between meals.', 'Sometimes', 'Frequently', 'Always'])
            smoking = st.radio("Do you smoke?* ", ['Yes', 'No'])
            water = st.radio("How often do you drink water daily?* ", ['I do not always drink water.', 'Sometimes', 'Frequently'])
            calories = st.radio("Do you often monitor your own calories?* ", ['Yes', 'No'])
            activity = st.radio("How often do you exercise?*", ['I do not exercise.', 'Sometimes', 'Frequently', 'Always'])
            technology = st.radio("How often do you use your electronic devices?* ", ['Sometimes', 'Frequently', 'Always'])
            alcohol = st.radio("Do you drink alcohol?* ", ['I do not drink alcohol.', 'Sometimes', 'Frequently', 'Always'])
            st.write("__NOTE:__ _Asterisk (*) represents the predictive attribute._")
            
            predict_button = st.form_submit_button('Predict')           
            
            if predict_button == True:
                # call the function to predict the outcomes for obesity level 
                obesity_pred_class, obesity_outcome_proba = obesity_predict(gender, 
                                                                            age, 
                                                                            height, 
                                                                            weight,
                                                                            family_history_obesity,
                                                                            caloric_food,
                                                                            fruit_vegetables,
                                                                            num_of_meals,
                                                                            between_meals,
                                                                            smoking,
                                                                            water,
                                                                            calories,
                                                                            activity,
                                                                            technology,
                                                                            alcohol)
                
                # call the function to display the obesity prediction outcomes
                obs_pred_outcome, obs_outcome_prob = display_obesity_outcome(name, obesity_pred_class, obesity_outcome_proba)
                
                # call function to insert user info and prediction outcomes into obesity database
                users_info = (name, identification_no, email, gender, age, height, weight, family_history_obesity, caloric_food, fruit_vegetables, num_of_meals, between_meals, smoking, water, calories, activity, technology, alcohol, int(obesity_pred_class), float(obs_outcome_prob), obs_pred_outcome)
                insert_obesity_pred_outcome_sql(DBFILENAME, users_info)
        
        
        
        
    # diabetes page
    if nav == 'Diabetes Mellitus':
        
        # display unique information about diabetes detection 
        st.sidebar.header("What's so unqiue about this diabetes prediction app? ")
        st.sidebar.write("This application uses __Gradient Boosting Algorithm__ (Supervised ML) with the best hyperparameters to predict the likelihood of developing diabetes in an individual. \
                            During the testing phase, the outcome shows that the gradient boosting algorithm can be achievable ___with at least 90% accuracy, precision, recall, and AUC.___ ")
        
        st.sidebar.write("_Evaluation outcome from the testing set_: ")
        diab_gbc_outcome_img = Image.open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\images\diab_gbc_outcome.jpg")
        st.sidebar.image(diab_gbc_outcome_img, output_format = "auto", width = 600)
        
        st.sidebar.write("__NOTE__: The objective of the testing set is to evaluate the performance of the trained model\
                            and was unseen during the training phase. ")
        st.sidebar.markdown("___")
        
        # display information on diabetes 
        st.sidebar.header("What's Diabetes? ")
        diab_sidebar_img = Image.open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\images\diab_sidebar_img.jpg")
        st.sidebar.image(diab_sidebar_img, output_format = "auto", width = 300)
        st.sidebar.write("__Diabetes__ is a chronic health condition that affects how your body generate food into energy.")
        st.sidebar.write("Most of the food you eat is broken down into sugar and released into your bloodstream. \
                            When your blood sugar elevated, it signals your pancreas to produce insulin. Insulin acts like \
                            a key to let the blood sugar into your body cells to use it as energy.")
        st.sidebar.write("If you have diabetes, your body either not producing enough insulin or cannot use the insulin it makes as well as it should.\
                            When there is not enough insulin or cell stop responding to insulin, too much blood sugar remains in your bloodstream. \
                            Overtime, that can cause serious health problems such as vision loss, or kidney disease, etc. ")
        st.sidebar.write("You may click the link below to know more about diabetes.(https://www.cdc.gov/diabetes/basics/diabetes.html)")
        st.sidebar.markdown("___")
        
        # display information about diabetes prevention
        st.sidebar.header("How can we control or prevent the chances of developing diabetes? ")
        diab_sidebar_img2 = Image.open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\images\diab_sidebar_img2.jpg")
        st.sidebar.image(diab_sidebar_img2, output_format = "auto", width = 300)
        st.sidebar.write( """
                             - Lose extra weight
                             - Be more physically active
                             - Eat healthy plant foods
                             - Eat healthier fats
                             - Stop smoking
                            """)
        st.sidebar.write("You may click the link below to know more about diabetes prevention and control.(https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/in-depth/diabetes-prevention/art-20047639)")
        st.sidebar.markdown("___")
        
        # bmi calculator 
        # call the function to compute the BMI
        calcBMI = bmi_calculation()
        # display the bmi value
        st.sidebar.info("Your BMI is {:.2f}".format(calcBMI))
        
        
        
        # add title to the diabetes page
        st.title("Diabetes Mellitus Prediction")
        
        # add header images on diabetes page
        diabetes_image = Image.open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\images\diabetes.jpg")
        st.image(diabetes_image, output_format = "auto", width = 600)
        
        st.info("__Medical Disclaimer__: This platform is not serve as an alternative to medical advice from medical professional healthcare provider.\
                 If you have any specific questions about any medical matter, you should consult your doctor or other medical professional healthcare provider. ")
        st.write("__General guideline(s) to the user:__")
        st.write("1) You are ___required___ to fill up all the information in this form in less than 5 mins. ")
        st.write("2) Some information may required your blood test results.")
        
        # prompt the user for input
        with st.form('Form 2'):
            name = st.text_input(label = "Enter your name:")
            identification_no = st.text_input(label = "Enter your identification no.:")
            email = st.text_input(label = "Enter your personal email address: ")
            age = st.number_input(label = "Enter your age: ", min_value = 0, max_value = 100, format = "%d")
            gender = st.radio('Gender:*', ['Male', 'Female'])
            bmi = st.number_input(label = "Enter your body mass index (BMI):*", min_value = 0.00, format = "%.2f", help = "BMI is a person's weight in kilograms divided by the square of height in metres.")
            glucose = st.number_input(label = "Enter the amount of fasting blood sugar level (in mmol/L):*", min_value = 0.00, format = "%.1f", help = "Blood sugar level is the measure of the concentration of glucose present in the human blood. The blood glucose level can be read from a blood glucose monitor device. ")
            st.write("__NOTE:__ _Asterisk (*) represents the predictive attribute._")
            
            # when the user clicks on the predict button 
            predict_button = st.form_submit_button('Predict')
            
            if predict_button == True:
                # call the function to predict the diabetes outcomes 
                diab_pred_class, diab_outcome_proba = diab_predict(gender, glucose, bmi)
                
                # call the function to display the diabetes prediction outcomes
                diab_pred_outcome_, diab_outcome_prob_ = display_diabetes_outcome(name, diab_pred_class, diab_outcome_proba)
                
                # call function to insert user info and prediction outcomes into diabetes database
                users_info = (name, identification_no, email, gender, age, bmi, glucose, int(diab_pred_class), float(diab_outcome_prob_), diab_pred_outcome_)
                insert_diabetes_pred_outcome_sql(DBFILENAME, users_info)
                
            
                
        
    # hypertension page
    if nav == 'Hypertension':
        
        # display unique information about hypertension detection 
        st.sidebar.header("What's so unqiue about this hypertension prediction app? ")
        st.sidebar.write("This application uses __Random Forest Algorithm__ (Supervised ML) with the best hyperparameters to predict the likelihood of developing hypertension in an individual. \
                            During the testing phase, the outcome shows that the random forest algorithm can be achievable ___with at least 90% accuracy, precision, recall, and AUC.___ ")
        
        st.sidebar.write("_Evaluation outcome from the testing set_: ")
        bp_etc_outcome_img = Image.open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\images\bp_rfc_outcome.jpg")
        st.sidebar.image(bp_etc_outcome_img, output_format = "auto", width = 600)
        
        st.sidebar.write("__NOTE__: The objective of the testing set is to evaluate the performance of the trained model\
                            and was unseen during the training phase. ")
        st.sidebar.markdown("___")
        
        
        # display information on hypertension 
        st.sidebar.header("What's Hypertension? ")
        bp_sidebar_img = Image.open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\images\bp_sidebar_img.jpg")
        st.sidebar.image(bp_sidebar_img, output_format = "auto", width = 300)
        st.sidebar.write("__Hypertension__ or __high blood pressure__ is a blood pressure that is higher than normal blood pressure. \
                            Your blood pressure changes throughout the day based on the level of activities. However, having blood pressure\
                            that is constantly above the normal level may result in the diagnosis of high blood pressure.")
        st.sidebar.write("High blood pressure usually develops over time and it can happen because of the unhealthy lifestyle choices\
                            or certain health conditions such as diabetes and having obesity, can also increase the chance of developing high blood pressure.")
        st.sidebar.write("You may click the link below to know more about hypertension.(https://www.cdc.gov/bloodpressure/about.htm)")
        st.sidebar.markdown("___")
        
        
        # display information about hypertension management and prevention
        st.sidebar.header("How can we manage and prevent the likelihood of developing high blood pressure? ")
        bp_sidebar_img2 = Image.open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\images\bp_sidebar_img2.jpg")
        st.sidebar.image(bp_sidebar_img2, output_format = "auto", width = 300)
        st.sidebar.write( """
                             - __Prevent High Blood Pressure__
                                 - Have a healthy diet
                                 - Maintain a healthy weight
                                 - Be physically active
                                 - Less smoking
                                 - Have adequate rest
                                 - Relax yourself
                            """)
        st.sidebar.write("You may click the link below to know more about prevention of high blood pressure.(https://www.cdc.gov/bloodpressure/prevent.htm)")
        st.sidebar.write( """
                             - __Management of High Blood Pressure__
                                 - Measure your blood pressure at the regular basis
                                 - Manage diabetes
                                 - Plan your lifestyle changes
                                 - Communicate with your medical professional healthcare provider
                            """)
        st.sidebar.write("You may click the link below to know more about management of high blood pressure.(https://www.cdc.gov/bloodpressure/manage.htm) ")
        st.sidebar.markdown("___")
        
        # bmi calculator 
        # call the function to compute the BMI
        calcBMI = bmi_calculation()
        # display the bmi value
        st.sidebar.info("Your BMI is {:.2f}".format(calcBMI))
        
        
        # add title
        st.title("Hypertension Prediction")
            
        # add header images on hypertension page
        bp_image = Image.open(r"C:\Users\jeffr\Desktop\20053371_WONG QI YUAN JEFFREY_C3879C_AY2021CWF\4. Others\images\hypertension.jpg")
        st.image(bp_image, output_format = "auto", width = 600)
        
        st.info("__Medical Disclaimer__: This platform is not serve as an alternative to medical advice from medical professional healthcare provider.\
                 If you have any specific questions about any medical matter, you should consult your doctor or other medical professional healthcare provider. ")
        st.write("__General instruction to the user:__")
        st.write("1) You are ___required___ to fill up all the information in this form in less than 5 mins. ")
        
        # prompt the user input
        with st.form('Form 3'):
            name = st.text_input(label = "Enter your name:")
            identification_no = st.text_input(label = "Enter your identification no.:")
            email = st.text_input(label = "Enter your personal email address: ")
            age = st.number_input(label = "Enter your age: ", min_value = 0, max_value = 100, format = "%d")
            gender = st.radio('Gender:*', ['Male', 'Female'])
            bmi = st.number_input(label = "Enter your body mass index (BMI):* ", min_value = 0.00, format = "%.2f", help = "BMI is a person's weight in kilograms divided by the square of height in metres.")
            sbp = st.number_input(label = "Enter the systolic blood pressure (in mm Hg):*", min_value = 0, format = "%d", help = "Systolic Blood Pressure is the top part of your blood pressure measurement as reflected on your blood pressure monitoring device.")
            dbp = st.number_input(label = "Enter the diastolic blood pressure (in mm Hg):*", min_value = 0, format = "%d", help = "Diastolic Blood Pressure is the bottom part of your blood pressure measurement as reflected on your blood pressure monitoring device.")
            st.write("__NOTE:__ _Asterisk (*) represents the predictive attribute._")
            
            # when the user clicks on the predict button 
            predict_button = st.form_submit_button('Predict')
            
            if predict_button == True:
                # call the function to predict outcomes for hypertension 
                bp_pred_class, bp_outcome_proba = hypertension_predict(bmi, dbp, sbp)
                
                # call the function to display hypertension prediction outcomes
                bp_pred_outcome_, bp_outcome_prob_ = display_hypertension_outcome(name, bp_pred_class, bp_outcome_proba)
                
                # call the function to insert user info and prediction outcomes into hypertension database
                users_info = (name, identification_no, email, gender, age, bmi, sbp, dbp, int(bp_pred_class), float(bp_outcome_prob_), bp_pred_outcome_)
                insert_hypertension_pred_outcome_sql(DBFILENAME, users_info)
                # insert all the users information and predicted outcomes into the SQL hypertension databases
                

        
if __name__ == '__main__':
    main()
