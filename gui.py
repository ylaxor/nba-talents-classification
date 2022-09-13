import streamlit as st
import joblib 
from argparse import ArgumentParser
import requests
import random 
import pandas as pd
from collections import Counter
from time import sleep
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import subprocess

def split_wisely(df, test_percent):
    ones = df[df['TARGET_5Yrs'] == 1]
    zeros = df[df['TARGET_5Yrs'] == 0]
    ones_test_samples_indices = random.sample(range(len(ones)), int(test_percent*len(ones)))
    zeros_test_samples_indices = random.sample(range(len(zeros)), int(test_percent*len(zeros)))
    ones_learn_samples_indices = list(set(range(len(ones))).difference(set(ones_test_samples_indices)))
    zeros_learn_samples_indices = list(set(range(len(zeros))).difference(set(zeros_test_samples_indices)))
    ones_learn_samples = ones.iloc[ones_learn_samples_indices]
    zeros_learn_samples = zeros.iloc[zeros_learn_samples_indices]
    learn_samples = pd.concat([ones_learn_samples, zeros_learn_samples], ignore_index=True)
    ones_test_samples = ones.iloc[ones_test_samples_indices]
    zeros_test_samples = zeros.iloc[zeros_test_samples_indices]
    test_samples = pd.concat([ones_test_samples, zeros_test_samples], ignore_index=True)
    return learn_samples, test_samples
def replace_nan(old_val, new_val0, new_val1, cls):
    if cls == 0:
        replace_with = new_val0
    else:
        replace_with = new_val1
    if np.isnan(old_val): return replace_with
    else: return old_val
def do_imputation(df, strategy):
    if strategy == "mean-by-class":
        new_val1 = df[df['TARGET_5Yrs'] == 1]['3P%'].mean()
        new_val0 = df[df['TARGET_5Yrs'] == 0]['3P%'].mean()
    elif strategy == "median-by-class":
        new_val1 = df[df['TARGET_5Yrs'] == 1]['3P%'].median()
        new_val0 = df[df['TARGET_5Yrs'] == 0]['3P%'].median()
    elif strategy == "mean-column":
        new_val1 = df['3P%'].mean()
        new_val0 = new_val1
    elif strategy == "median-column":
        new_val1 = df['3P%'].median()
        new_val0 = new_val1
    else:
        new_val1 = 0.0
        new_val0 = new_val1
    
    df['new_3P%'] = df.apply(lambda x: replace_nan(old_val=x['3P%'], new_val0=new_val0, new_val1=new_val1, cls=int(x['TARGET_5Yrs'])), axis=1)
    df = df.drop('3P%', axis=1)
    df.rename({'new_3P%':'3P%'}, inplace=True, axis=1)
    return df
if 'learning_dataframe' not in st.session_state:
    st.session_state['learning_dataframe'] = pd.read_csv("./dataframe.csv")
if 'custom_learning_dataframe' not in st.session_state:
    st.session_state['custom_learning_dataframe'] = []
if 'output_dir' not in st.session_state:
    st.session_state["output_dir"] = "./assets"
    Path(st.session_state["output_dir"]).mkdir(parents=True, exist_ok=True)
st.markdown("# 🏀 NBA Talents")
#predictors_names = joblib.load(args.folder+'/'+args.predictors) 
st.markdown('**A beginner NBA player is worth investing in if based on his sports statistics, he would likely stay in the NBA for more than 5 years.**')

menu = ["Home", "Dataset", "Model training", "Performance", "Inference"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.write("Use the menu in the sidebar to see how we developed a classification machine learning model to help investors make alike decision. It predicts the probability that a player would stay in NBA for more than 5 years, based on some relevant sports indicators in the NBA.")
elif choice == "Dataset":
    df = st.session_state['learning_dataframe']
    with st.expander("Training dataframe"):
        cols = st.columns(3)
        with cols[0]:
            st.write("Players")
            st.dataframe(df['Name'])
        with cols[1]:
            st.write("Predictors")
            st.write(df.drop(['Name', 'TARGET_5Yrs'], axis=1))
        with cols[2]:
            st.write("Targets")
            st.dataframe(df['TARGET_5Yrs'])

    with st.expander("Predictors description"):
        st.dataframe(df.drop(['Name', 'TARGET_5Yrs'], axis=1).describe())
    with st.expander("Target description"):
        st.write(Counter(df['TARGET_5Yrs']))

    with st.expander("Santiy of predictors"):
        cols = st.columns(2)
        with cols[0]:
            st.write("Missing/ NaN values")
            nans = pd.DataFrame(df.isna().sum(), columns=["count_nans"])
            st.write(nans[nans["count_nans"]>0])
        with cols[1]:
            st.write("Redundancy among predictors")
            st.markdown("$$FT\% \sim FTM/FTA$$")
            st.markdown("$$FG\% \sim FGM/FGA$$")
            st.markdown("$$3P\% \sim 3PM/3PA$$")
            st.markdown("$$REB \sim OREB + DREB$$")
    with st.expander("Santiy of target variable"):
        cols = st.columns(2)
        with cols[0]:
            st.write("Missing/ NaN values")
            st.write(df['TARGET_5Yrs'].isna().sum())
        with cols[1]:
            st.write("Labels imbalance")
            st.write(Counter(df['TARGET_5Yrs']))

elif choice == "Model training":
    st.markdown("Upload data file, and retrain the best found target classifier on it.")
    with st.expander("Training data file"):
        file = st.file_uploader("Select dataframe in csv format")
        if st.button("Use as training data"):
            if file:
                try:
                    st.session_state['custom_learning_dataframe'] = pd.read_csv(file)
                    st.success("Datafile was uploaded correctly and is shown below. It can now be used to train the best model.")
                    st.write(st.session_state['custom_learning_dataframe'])
                except:
                    st.warning("Wrong csv file.")
            else:
                st.warning("Please select a file first.")
    st.write("The best chosen model and best learning settings were:")
    st.write("* SVM. With Polynomial kernel, which means that players data was not linearly separable and it contains non-linearities. The degree of the polynomial kernel used in the SVM was 4. The (inverse) regularization parameter C of the model was 1.0")
    st.write("* Before hitting the SVM, our data points were normalized using the MinMax scaler (was compared to StandardScaling). The dimensionality of normalized data points was then reduced using a Recursive Feature Selection process. We kept only 11 features in the final best model.")
    st.markdown("Please head to [this notebook](https://github.com/ylaxor/nba-talents-classification/blob/main/dev.ipynb) for details about the selection process of the best model.")
    
    with st.expander("Classifier model parameters"):
        c_param = st.slider("C", 1e-1, 2.0)
        d_param = st.slider("degree of polynom", 1, 5)
        if st.button("Retrain SVM on the above data"):
            df = st.session_state['custom_learning_dataframe']
            if not isinstance(df, list):
                df = do_imputation(df, strategy="median-by-class")
                df.drop('OREB', axis=1, inplace=True)
                df.drop('DREB', axis=1, inplace=True)
                df.drop('FTM', axis=1, inplace=True)
                df.drop('FTA', axis=1, inplace=True)
                df.drop('FGM', axis=1, inplace=True)
                df.drop('FGA', axis=1, inplace=True)
                df.drop('3P Made', axis=1, inplace=True)
                df.drop('3PA', axis=1, inplace=True)
                features_names = df.columns.values.tolist()
                if 'Name' in features_names: features_names.remove('Name')
                if 'TARGET_5Yrs' in features_names: features_names.remove('TARGET_5Yrs')
                st.write("Used {} data features after application of RFE: {}".format(len(features_names), "-".join(features_names)))
                learn_df, test_df = split_wisely(df, test_percent=0.15)
                Xlearn = learn_df.drop(['Name', 'TARGET_5Yrs'], axis=1)
                Ylearn = learn_df['TARGET_5Yrs']
                Xtest = test_df.drop(['Name', 'TARGET_5Yrs'], axis=1)
                Ytest = test_df['TARGET_5Yrs']
                normalizer = MinMaxScaler()
                normalizer.fit(Xlearn.to_numpy())
                Xlearn_normalized = normalizer.transform(Xlearn)
                model = SVC(random_state=1, C=c_param, kernel="poly", degree=d_param)
                transformer = RFE(estimator=SVC(random_state=1, kernel="linear", C=1), n_features_to_select=11, step=1)
                X_learn_reduced = transformer.fit_transform(Xlearn_normalized, Ylearn)
                model.fit(X_learn_reduced, Ylearn)
                st.success("Finished training and saving best model files to local storage. Now you can see how this model performs on some testing chunk.")
                joblib.dump(features_names, st.session_state['output_dir'] + '/' + "features.save") 
                joblib.dump(normalizer, st.session_state['output_dir'] + '/' + "normalizer.save") 
                joblib.dump(transformer, st.session_state['output_dir'] + '/' + "transformer.save")
                joblib.dump(model, st.session_state['output_dir'] + '/' + "model.save")
                joblib.dump(test_df, st.session_state['output_dir'] + '/' + "testframe.save")

elif choice == "Performance":
    st.write("Here we test the pre-trained model on the corresponding testing dataframe. We use confusion matrix and f1 measures to assess the quality of our classification task.")
    test_df = joblib.load(st.session_state['output_dir'] + '/' + 'testframe.save') 
    Xtest = test_df.drop(['Name', 'TARGET_5Yrs'], axis=1)
    Ytest = test_df['TARGET_5Yrs']
    normalizer = joblib.load(st.session_state['output_dir'] + '/' + 'normalizer.save') 
    transformer = joblib.load(st.session_state['output_dir'] + '/' + 'transformer.save') 
    model = joblib.load(st.session_state['output_dir'] + '/' + 'model.save') 
    Xtest_normalized = normalizer.transform(Xtest)
    Xtest_featurized = transformer.transform(Xtest_normalized)
    predicted_test_labels = model.predict(Xtest_featurized)
    test_confusion_mat = confusion_matrix(Ytest, predicted_test_labels)
    test_recall = recall_score(Ytest, predicted_test_labels, average="macro")
    test_f1 = f1_score(Ytest, predicted_test_labels, average="macro")
    cols = st.columns(3)
    with cols[0]:
        st.write("Test set contains")
        st.write(Counter(Ytest))
    with cols[1]:
        st.write("Confusion matrix", test_confusion_mat)
    with cols[2]:
        st.write("macro-average recall score {:.4f}".format(test_recall))
        st.write("macro-average f1 score {:.4f}".format(test_f1))

elif choice == "Inference":
    launch_api = st.button("launch (flask) API server.", key="23")
    if launch_api:        
        subprocess.Popen(['python', 'api.py'])

        
    digit2class = {1: "Good invest.", 0: "Not Good invest."}
    api_server = 'http://127.0.0.1:8080/predict'

    try:
        predictors_names = joblib.load(st.session_state['output_dir']+'/'+'features.save') 

        api_server = st.text_input("API server url", api_server)
        st.write('Please set values for the below sports parameters in order to:')
        
        predict_btn = st.button("Infer whether the NBA ⛹️‍♂️, ⛹️‍♀️ in question is a good invest, or not.")
        sliders = [st.empty() for p in predictors_names]
        vals = []
        result_holder = st.empty()
            
        f = len(predictors_names) // 4
        s = 0
        for x in range(f):
            cols = st.columns(4)
            with cols[0]:
                p = predictors_names[s]
                b = st.slider('{}'.format(p), 0.0, 100.0, 0.1, key=s)
                sliders[x] = b
                vals.append(b)
            with cols[1]:
                p = predictors_names[s+1]
                b = st.slider('{}'.format(p), 0.0, 100.0, 0.1, key=s+1)
                sliders[x] = b
                vals.append(b)
            with cols[2]:
                p = predictors_names[s+2]
                b = st.slider('{}'.format(p), 0.0, 100.0, 0.1, key=s+2)
                sliders[x] = b
                vals.append(b)
            with cols[3]:
                p = predictors_names[s+3]
                b = st.slider('{}'.format(p), 0.0, 100.0, 0.1, key=s+3)
                sliders[x] = b
                vals.append(b)
            s += 4
        if len(predictors_names) % 4 != 0:
            remaining_names = predictors_names[f*4:]
            cols = st.columns(4)
            for i, r in enumerate(remaining_names):
                with cols[i]:
                    p = predictors_names[f*4+i]
                    b = st.slider('{}'.format(p), 0.0, 100.0, 0.1, key=f*4+i)
                    sliders[x] = b
                    vals.append(b)
                    
        if predict_btn:
            input_dict = {k:v for k,v in zip(predictors_names, [x for x in vals])}
            try:
                r = requests.get(api_server, params=input_dict)
                try:
                    cls = int(eval(r.text)['prediction'][0])
                    if cls == 1:
                        result_holder.success("Given the stats. below, this player is likely {}".format(digit2class[cls].lower()))
                    else:
                        result_holder.error("Given the stats. below, this player is likely {}".format(digit2class[cls].lower()))  
                except:
                    result_holder.warning("{}".format(list(eval(r.text)['error'])[0]))
            except:
                result_holder.info("API server at {} is down.".format(api_server)) 
    except:
        st.write("Can't load list of parameters. Please verify the given folder and filename params.")
