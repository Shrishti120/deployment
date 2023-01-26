import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy.sparse import hstack
import pickle as cPickle
import yaml

## loading config file 
config = yaml.load(open('config/config.yaml', 'r'), Loader=yaml.FullLoader)


def load_vectors(file_name):
    # save the classifier
    with open(f"{file_name}", 'rb') as fid:
        vector = cPickle.load(fid)
    return vector

def stack_all_values(dataset):
    ##Loading the data
    data = pd.read_csv(dataset)
    bow_vectorize_train = load_vectors(config['VECTORS']['BOW'])
    vectorizer_school_state = load_vectors(config['VECTORS']['SCHOOL_STATE'])
    vectorizer_teacher_prefix = load_vectors(config['VECTORS']['TEACHER_PREFIX'])
    vectorizer_clean_categories = load_vectors(config['VECTORS']['CLEAN_CATEGORIES'])
    vectorizer_clean_subcategories = load_vectors(config['VECTORS']['CLEAN_SUBCATEGORIES'])
    vectorizer_project_grade_category = load_vectors(config['VECTORS']['PROJECT_GRADE_CATEGORY'])
   
    # print(data)

    # Creating the approved and rejected data frames

    project_approved = data[data['project_is_approved'] == 1]
    project_reject = data[data['project_is_approved'] == 0]


    # Reference: https://stackoverflow.com/questions/51835369/combine-two-dataframes-one-row-from-each-one-at-a-time-python-pandas
    data = pd.concat([project_approved, project_reject]).sort_index(kind='merge')

    ## Here we are dividing our data into two parts 1. Where our project is approved 2. Where our project is not approved
    y = data['project_is_approved'].values
    x = data.drop(['project_is_approved'], axis=1)


    ## 2. Transforming the data to its vectors form 
    bow_data = bow_vectorize_train.transform(data['essay'])

    #3 School State 

    school_state = vectorizer_school_state.transform(data['school_state'].values)
     
    ## Teacher_prefix

    teacher_prefix = vectorizer_teacher_prefix.transform(data['teacher_prefix'].values)

    # Clean Category 

    clean_categories = vectorizer_clean_categories.transform(data['clean_categories'].values)
        
    # Clean Sub Category

    clean_subcategories = vectorizer_clean_subcategories.transform(data['clean_subcategories'].values)

        

    # Project Grade Category 

    project_grade_category = vectorizer_project_grade_category.transform(data['project_grade_category'].values)
    ## 1.4.2 Performing Vectorisation on Numerical Data

 
    # 1.5 CONCATINATING ALL FEATURES USING hstack()

    x_concat = hstack((bow_data,school_state , teacher_prefix , clean_categories , clean_subcategories , project_grade_category ))
    x_concat.toarray()
    dense_array = x_concat.todense()
    my_array = np.asarray(dense_array)


    return x_concat , y


def multnomial_nb(path):

    x_concat  , y = stack_all_values(path)

    # load it again
    with open(config['MODEL']['NB'], 'rb') as fid:
        multnomial_nb = cPickle.load(fid)

    predicted_value = multnomial_nb.predict(x_concat)
    predicted_value = predicted_value.tolist()
    predicted_value_new = [] 
    for i in predicted_value:
        if i == 1:
            predicted_value_new.append("Project Accepted")
        else:
            predicted_value_new.append('Project Not Accepted')

    score = multnomial_nb.score(x_concat,y)
    accuracy = score * 100
    return f"{accuracy}%"  , predicted_value_new


def logixtic_regression(path):
    x_concat  , y = stack_all_values(path)
    # load it again
    with open(config['MODEL']['LR'], 'rb') as fid:
        logistic_regression = cPickle.load(fid)

        predicted_value = logistic_regression.predict(x_concat)
        predicted_value = predicted_value.tolist()
        predicted_value_new = [] 
        for i in predicted_value:
            if i == 1:
                predicted_value_new.append("Project Accepted")
            else:
                predicted_value_new.append('Project Not Accepted')

    score = logistic_regression.score(x_concat,y)
    accuracy = score * 100
    return f"{accuracy}%"  , predicted_value_new


