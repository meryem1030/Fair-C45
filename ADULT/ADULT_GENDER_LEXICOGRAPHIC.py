#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import random
import numpy as np
import pandas as pd
from statistics import mode
from collections import Counter
import sklearn
import sklearn.datasets
import sklearn.ensemble
from sklearn import preprocessing
from scipy.stats import rankdata
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from timeit import default_timer as timer
start = timer()


# In[2]:


# minimum difference between continuous values
DELTA = 1e-5

# minumum number of instances for each node
MINIMUM_INSTANCES = 2

# limit for the minimum number of instances in a continuous interval
INTERVAL_LIMIT = 25

# correction value used in the threshold and error calculation (based on C4.5)
EPSILON = 1e-3

# precision correction
CORRECTION = 1e-10

# values below this are considered zero for gain ratio calculation
ZERO = 1e-7

# make sure it is not trying to modify a copy of a dataframe
pd.options.mode.chained_assignment = 'raise'

# sets a fixed random seed
random.seed(1000)

# (dis/en)able the pruning
PRUNE = False


# In[3]:


print('GENDER LEXICOGRAPHIC')





def load_csv(path):

    data = pd.read_csv(path)
    
    data['Income'] = np.where(data['Income'] == '>50K', 1, 0)
    for col in data.columns:
        data[col] = data[col].replace(r'?', np.nan)
    df = data
        
    categorical_features = []    
    for i in range(len(data.iloc[:,:-1].columns)):
        if is_numeric_dtype(data.iloc[:,i]) == False:
            categorical_features.append(i)
        
    # attribute metadata information
    metadata_original = {}
    
    for index, attribute in enumerate(data.notna().columns):
        data_domain = []
        if is_numeric_dtype(data[attribute]): #
            data_domain.append(data[attribute].min())
            data_domain.append(data[attribute].max())
        else:
            data_domain = data.loc[data[attribute].notna(), attribute].unique()
            

        metadata_original[attribute] = data_domain
            
    return df,categorical_features, metadata_original, data
df,categorical_features, metadata_original, data = load_csv(r'adult.csv')
df_test,categorical_features_test, metadata_original_test, data_test = load_csv(r'adult_test.csv')





def convert_categories(data, categorical_features):
    count = 0
    for col_idx in categorical_features:
        unique_val = data.iloc[data.iloc[:, col_idx].notna().index, col_idx].unique()
        for u in range(len(unique_val)):
            for i in range(len(data)):
                if data.iloc[i,col_idx] == unique_val[u]:
                    data.iloc[i,col_idx] = u + 1
    return data
data_test = convert_categories(data_test, categorical_features_test)   
data = convert_categories(data, categorical_features)         





def numerical_attributes(data, categorical_features):
    numerical_names = []
    for col_idx in range(len(data[0,:-1])):
        if col_idx not in categorical_features:
            numerical_names.append(col_idx)
    return numerical_names





merged_data = data
for i in range(len(data_test)):
    merged_data = merged_data.append(data_test.iloc[i,:], ignore_index=True)





merged_data = merged_data.to_numpy()
data = data.to_numpy()
data_test = data_test.to_numpy()
numerical_names = numerical_attributes(data, categorical_features)
merged_data = merged_data.astype(float)
data = data.astype(float)
data_test = data_test.astype(float)





def convert_matadata(data, numerical_names):

    metadata = {}

    for index in range(np.shape(data)[1]): 
        data_domain = []
        if index in numerical_names: #
            data_domain.append(min(data[:,index]))
            data_domain.append(max(data[:,index]))
        else:
            data_domain = np.unique(data[~np.isnan(data[:, index]), index])


        metadata[index] = data_domain
    return metadata

metadata = convert_matadata(merged_data, numerical_names)





def gain_ratio(data, index, weights):
    # find unique values except nan in the class
    labels = np.unique(data[:,-1][~np.isnan(data[:,index])])
    valid_data = data[~np.isnan(data[:,index])]
    S = []
    for l in labels.tolist():
        
        S.append(weights[np.where(valid_data[:,-1] == l)].sum())
    lenght = weights.sum()
    missing_lenght = weights[np.isnan(data[:,index])].sum()
    known_lenght = lenght - missing_lenght
    weights_valid = np.ones(len(valid_data))

    total_entropy = 0
    #calculation of entropy on class label
    for s in S:
        total_entropy -=((s / known_lenght) * np.log2(s / known_lenght))
    values = np.unique(valid_data[:,index][~np.isnan(valid_data[:,index])])
    outer = 0
    val_split = 0
    for val in values.tolist():
        inner = 0
        val_len = weights_valid[valid_data[:,index] == val].sum() 
        for l in labels.tolist():
            #how many val is assigned to the class l ---> count of values assigned the label in class
            a =np.where(valid_data[:,index] == val)[0].tolist()
            b = np.where(valid_data[:,-1]==l)[0].tolist()  
            cvl = weights[list(set(a).intersection(b))].sum()
            
            if cvl == 0:
                inner -= 0
            else:
                inner -= (cvl/val_len) * np.log2(cvl/val_len)
        if val_len == 0:
            outer += 0
        else:
            outer += (val_len/known_lenght)*inner
        split = val_len / (known_lenght + missing_lenght)
        val_split -= split * np.log2(split)

    if missing_lenght > 0:
        m = missing_lenght / (known_lenght + missing_lenght)
        val_split -= m * np.log2(m)
    gain = (known_lenght / (known_lenght + missing_lenght)) * (total_entropy - outer)
    gain_ratio = 0 if gain == 0 else gain / val_split
    return gain_ratio, gain, val_split





def candidate_thresholds(data, index, weights):
    valid = list(~np.isnan(data[:,index]))
    values = list(zip(np.array(data[valid, index]), weights[valid]))
    values.sort()

    length = weights[valid].sum()
    interval_length = 0
    thresholds = []

    # minimum number of instances per interval (according to C4.5)
    class_values = list(np.unique(data[:,-1][~np.isnan(data[:,-1])]))
    min_split = 0.1 * (length / len(class_values))

    if min_split <= MINIMUM_INSTANCES:
        min_split = MINIMUM_INSTANCES
    elif min_split > INTERVAL_LIMIT:
        min_split = INTERVAL_LIMIT

    for s in range(len(values) - 1):
        interval_length += values[s][1]
        length -= values[s][1]
        if (values[s][0] + DELTA) < values[s + 1][0] and (interval_length + CORRECTION) >= min_split and (length + CORRECTION) >= min_split:
            thresholds.append((values[s][0] + values[s + 1][0]) / 2)
    return thresholds





def gain_ratio_numeric(data, index, weights):
    thresholds = candidate_thresholds(data, index, weights)
    if len(thresholds) == 0:
        return 0, 0, 0
    valid_data = data[~np.isnan(data[:,index])].copy()
    valid_weights = np.ones(len(valid_data))
    values = valid_data[:, index].copy()

    length = valid_weights.sum()   # sum of instances with known outcome

    gain_correction = length / weights.sum()
    penalty = np.log2(len(thresholds)) / weights.sum()
    gain_information = []
    for t in thresholds:
        valid_data = data[~np.isnan(data[:,index])].copy() # copying/masking data, each time new data to use binary val change
        valid_data[:,index] = np.where(valid_data[:,index] > t, 1.0, 0.0)

        _, gain, split = gain_ratio(valid_data, index, valid_weights) # 3 is index

        # apply a penalty for evaluating multiple threshold values (based on C4.5)
        gain = (gain_correction * gain) - penalty
        ratio = 0 if gain == 0 or split == 0 else gain / split

        gain_information.append((ratio, gain))

        thresholds_gain = [g[1] for g in gain_information]
        selected = np.argmax(thresholds_gain)
    
    return gain_information[selected][0], gain_information[selected][1], thresholds[selected]


# In[ ]:


def available_attribute(data):
    available = np.ones((data.shape[1]-1), dtype=bool)
    return list(available)






def split_DI_CV(data, pro_index, categorical_features, pri_val, unpri_val, pri_class):
    data_original = data.copy()
    weights = np.ones(len(data_original))
    if sum(pd.isna(data_original[:,pro_index])) > 0:
        lenght = weights.sum()
        number_of_missing = weights[pd.isna(data_original[:,pro_index])].sum()
        missing_positive = sum(data[pd.isna(data[:, pro_index]), :][:,-1] == pri_class)
        known_length = lenght - number_of_missing
        known_lenght_positive = sum(data[pd.notna(data[:, pro_index]), :][:,-1] == pri_class)
    else: 
        lenght = weights.sum()
        number_of_missing = 0
        missing_positive = 0
        known_length = lenght
        known_lenght_positive = sum(data[pd.notna(data[:, pro_index]), :][:,-1] == pri_class)
        
    if pro_index in categorical_features: #transform in binary case if the index in categorical list
        unique_values = np.unique(data_original[:,pro_index][~pd.isna(data_original[:,pro_index])])
        data_original[(pd.notna(data_original[:,pro_index]) & (data_original[:,pro_index] != pri_val)), pro_index] = unpri_val
    else:# 1 is prival, 0 is unpri_val after changing the continues values to the binary
        att_threshold = gain_ratio_numeric(data_original, pro_index, weights)[2]
        data_original[pd.notna(data_original[:,pro_index]), pro_index] = np.where(data_original[pd.notna(data_original[:,pro_index]), pro_index] > att_threshold, pri_val, unpri_val)
    unique_values = np.unique(data_original[:,pro_index][~pd.isna(data_original[:,pro_index])]).tolist()
    try:
        pri_data = data_original[data_original[:,pro_index] == pri_val, :] 
    except:
        pri_data = []
    try:
        unpri_data = data_original[data_original[:,pro_index] == unpri_val, :]
    except:
        unpri_data = []

    #number of priviliged values that assigned to positive class 
    #so, I used pri_data to reach priviliged values that are assigned to positive class
    try:
        privileged = sum(pri_data[:,-1] == pri_class) 
    except:
        privileged = 0
    try:
        lenght_pri = len(pri_data[:,pro_index])  
    except:
        lenght_pri = 0
    if missing_positive == 0 or known_lenght_positive == 0:
        F_privileged = 0
    else:
        F_privileged = missing_positive * (privileged/known_lenght_positive)   
    if privileged == 0 or lenght_pri == 0:
        privileged_probability = 0
    else:
        privileged_probability = (privileged + F_privileged) / (lenght_pri + number_of_missing)
        
    try:
        unprivileged = sum(unpri_data[:,-1] == pri_class)
    except:
        unprivileged = 0
    try:
        lenght_unpri = len(unpri_data[:,pro_index])
    except:
        lenght_unpri = 0 
    if missing_positive == 0 or known_lenght_positive == 0:
        F_unprivileged = 0
    else:
        F_unprivileged = missing_positive * (unprivileged/known_lenght_positive)
    if unprivileged == 0 or lenght_unpri == 0:
        unprivileged_probability = 0
    else:
        unprivileged_probability = (unprivileged + F_unprivileged) / (lenght_unpri + number_of_missing)
        
    if unprivileged_probability == 0 or privileged_probability == 0:
        DI_DATA = 0
    else:
        DI_DATA = unprivileged_probability / privileged_probability
        
    try:    
        CV_DATA = privileged_probability - unprivileged_probability
    except:
        CV_DATA = 0
        
    return pri_data, unpri_data, DI_DATA, CV_DATA       





from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
def consistency(data_split, categorical_features, y_split, pro_index, k=5):
    if len(data_split) < k:
        k = len(data_split)
    cols = []
    if len(data_split) > 0:
        [cols.append(i) for i in categorical_features if i!=pro_index]
        ct = ColumnTransformer([("encoder", OneHotEncoder(), cols)], remainder = 'passthrough')
        
        data_split = ct.fit_transform(data_split)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data_split)
        indices = nbrs.kneighbors(data_split, return_distance=False)
        return 1 - abs(y_split - y_split[indices].mean(axis=1)).mean()
    else:
        data_split = 0
        y_split = 0
        return 0
def consistency_data(data, pro_index, categorical_features, pri_val, unpri_val, pri_class):
    weights = np.ones(len(data))
    try:
        pri_data = split_DI_CV(data, pro_index, categorical_features, pri_val, unpri_val, pri_class)[0]
        X_priviliged = pri_data[~np.isnan(pri_data).any(axis=1)][:,:-1]
        y_priviliged = pri_data[~np.isnan(pri_data).any(axis=1)][:,-1]
    except:
        pri_data = []
        X_priviliged = []
        y_priviliged = []
    consistency_pri = consistency(X_priviliged, categorical_features, y_priviliged, pro_index, k=5)
    try:
        unpri_data = split_DI_CV(data, pro_index, categorical_features, pri_val, unpri_val, pri_class)[1]
        X_unpriviliged = unpri_data[~np.isnan(unpri_data).any(axis=1)][:,:-1]
        y_unpriviliged = unpri_data[~np.isnan(unpri_data).any(axis=1)][:,-1]
    except:
        unpri_data = []
        X_unpriviliged = []
        y_unpriviliged = []
    consistency_unpri = consistency(X_unpriviliged, categorical_features, y_unpriviliged, pro_index, k=5)
    return consistency_pri + consistency_unpri





def fairness_attribute(data, categorical_features, pro_index, index_att, pri_val, unpri_val, pri_class):
    weights = np.ones(len(data))
    data_original = data.copy()
    unique_list = []
    if index_att in categorical_features:
        unique_values = np.unique(data_original[:,index_att][~pd.isna(data_original[:,index_att])])
    else:
        att_threshold = gain_ratio_numeric(data_original, index_att, weights)[2]
        data_original[pd.notna(data_original[:,index_att]), index_att] = np.where(data_original[pd.notna(data_original[:,index_att]), index_att] > att_threshold, 1, 0)
        unique_values = np.unique(data_original[:,index_att][~pd.isna(data_original[:,index_att])]).tolist()  
    DI_results = []
    CV_results = []
    Cons_result = []
    for u in unique_values:
        unique_list.append(u)
        #filter the data according to unique values of the attribute
        print('unique_value:', u)
        filter_data = data_original[data_original[:,index_att] == u, :]
        fairness = split_DI_CV(filter_data, pro_index, categorical_features, pri_val, unpri_val, pri_class)
        cons = consistency_data(filter_data, pro_index, categorical_features, pri_val, unpri_val, pri_class)
        DI_results.append(fairness[2])
        CV_results.append(fairness[3])
        Cons_result.append(cons)
        
    return sum(DI_results), sum(CV_results), sum(Cons_result)






def fairness_diffs(data, categorical_features, available_attributes, pro_index, pri_val, unpri_val, pri_class):
    DIs = []
    CVs = []
    consistencies = []
    ind = []
    predictors = available_attributes
    for i in range(len(predictors)):
        if np.unique(data[:,i][~pd.isna(data[:,i])]).tolist() != []:
            if i != pro_index and predictors[i] == True:
                ind.append(i)
                f_attribute = fairness_attribute(data, categorical_features, pro_index, i, pri_val, unpri_val, pri_class)
                f_data = split_DI_CV(data, pro_index, categorical_features, pri_val, unpri_val, pri_class)
                DI = f_attribute[0] - f_data[2]
                DIs.append(DI)
                CV = f_data[3] - f_attribute[1]
                CVs.append(CV)
                cons = f_attribute[2] - consistency_data(data, pro_index, categorical_features, pri_val, unpri_val, pri_class)
                consistencies.append(cons)
            if i == pro_index and predictors[i] == True:
                f_data = split_DI_CV(data, pro_index, categorical_features, pri_val, unpri_val, pri_class)
                ind.append(i)
                DI = f_data[2]
                DIs.append(DI)
                CV = f_data[3]
                CVs.append(CV)
                cons = consistency_data(data, pro_index, categorical_features, pri_val, unpri_val, pri_class)
                consistencies.append(cons)
        else:
            ind.append(i)
            DI = 0
            DIs.append(DI)
            CV =  -1
            CVs.append(CV)
            cons = 0
            consistencies.append(cons)
    return ind, DIs, CVs, consistencies






def DT_LEFT(data, pro_index, index_att, available_attributes, categorical_features, pri_val, unpri_val, pri_class):
    weights = np.ones(len(data))
    PRI = split_DI_CV(data, pro_index, categorical_features, pri_val, unpri_val, pri_class)[0]
    UNPRI = split_DI_CV(data, pro_index, categorical_features, pri_val, unpri_val, pri_class)[1]
    weights_pri = np.ones(len(PRI))
    weights_unpri = np.ones(len(UNPRI))
    data_original = data.copy()
    if sum(pd.isna(data_original[:,index_att])) > 0:
        number_missing_pri = weights_pri[pd.isna(PRI[:,index_att])].sum()
        known_lenght_pri = weights_pri.sum() - number_missing_pri
        missing_positive_pri = sum(PRI[pd.isna(PRI[:, index_att]), :][:,-1] == pri_class)
        known_lenght_positive_pri = sum(PRI[pd.notna(PRI[:, index_att]), :][:,-1] == pri_class)
        
        number_missing_unpri = weights_unpri[pd.isna(UNPRI[:,index_att])].sum()
        known_lenght_unpri = weights_unpri.sum() - number_missing_unpri
        missing_positive_unpri = sum(UNPRI[pd.isna(UNPRI[:, index_att]), :][:,-1] == pri_class)
        known_lenght_positive_unpri = sum(UNPRI[pd.notna(UNPRI[:, index_att]), :][:,-1] == pri_class)
    else:
        number_missing_pri = 0
        known_lenght_pri = weights_pri.sum() - number_missing_pri
        missing_positive_pri = 0
        known_lenght_positive_pri = sum(PRI[pd.notna(PRI[:, index_att]), :][:,-1] == pri_class)
        
        number_missing_unpri = 0
        known_lenght_unpri = weights_unpri.sum()
        missing_positive_unpri = 0
        known_lenght_positive_unpri = sum(UNPRI[pd.notna(UNPRI[:, index_att]), :][:,-1] == pri_class)
    if index_att in categorical_features:
        unique_values = np.unique(data_original[:,index_att][~pd.isna(data_original[:,index_att])])
    else:
        att_threshold = gain_ratio_numeric(data_original, index_att, weights)[2]
        data_original[pd.notna(data_original[:,index_att]), index_att] = np.where(data_original[pd.notna(data_original[:,index_att]), index_att] > att_threshold, 1, 0)
        PRI[pd.notna(PRI[:,index_att]), index_att] = np.where(PRI[pd.notna(PRI[:,index_att]), index_att] > att_threshold, 1, 0)
        UNPRI[pd.notna(UNPRI[:,index_att]), index_att] = np.where(UNPRI[pd.notna(UNPRI[:,index_att]), index_att] > att_threshold, 1, 0)
        unique_values = np.unique(data_original[:,index_att][~pd.isna(data_original[:,index_att])]).tolist()  
    unique_list = []
    probs_privals = []
    probs_unprivals = []
    for u in unique_values:
        unique_list.append(u)
        try:
            filter_data_priviliged = PRI[PRI[:,index_att] == u, :]
        except:
            filter_data_priviliged = []
        try:
            filter_data_unpriviliged = UNPRI[UNPRI[:,index_att] == u, :]
        except:
            filter_data_unpriviliged = []
        try:
            result_pri = sum(filter_data_priviliged[:,-1] == pri_class)
        except:
            result_pri = 0
        try:
            lenght_pri = len(filter_data_priviliged[:,index_att])
        except:
            lenght_pri = 0
        if missing_positive_pri == 0 or known_lenght_positive_pri == 0:
            F_pri = 0
        else:
            F_pri = missing_positive_pri * (result_pri/known_lenght_positive_pri)   
        if result_pri == 0 or lenght_pri == 0:
            prob_pri = 0
        else:
            prob_pri = (result_pri + F_pri) / (lenght_pri + number_missing_pri)
        probs_privals.append(prob_pri)
        
        
        try:
            result_unpri = sum(filter_data_unpriviliged[:,-1] == pri_class)
        except:
            result_unpri = 0
        try:
            lenght_unpri = len(filter_data_unpriviliged[:,index_att])
        except:
            lenght_unpri = 0
        if missing_positive_unpri == 0 or known_lenght_positive_unpri == 0:
            F_unpri = 0
        else:
            F_unpri = missing_positive_unpri * (result_unpri/known_lenght_positive_unpri)   
        if result_unpri == 0 or lenght_unpri == 0:
            prob_unpri = 0
        else:
            prob_unpri = (result_unpri + F_unpri) / (lenght_unpri + number_missing_unpri)
        probs_unprivals.append(prob_unpri)
        
    return sum(probs_privals) + sum(probs_unprivals)
            
            
def DT_RIGHT(data, index_att, available_attributes, categorical_features, pri_class):
    data_original = data.copy()
    weights = np.ones(len(data))
    if sum(pd.isna(data_original[:,index_att])) > 0:
        lenght = weights.sum()
        number_of_missing = weights[pd.isna(data_original[:,index_att])].sum()
        missing_positive = sum(data_original[pd.isna(data_original[:, index_att]), :][:,-1] == pri_class)
        known_length = lenght - number_of_missing
        known_lenght_positive = sum(data_original[pd.notna(data_original[:, index_att]), :][:,-1] == pri_class)
    else: 
        lenght = weights.sum()
        number_of_missing = 0
        missing_positive = 0
        known_length = lenght
        known_lenght_positive = sum(data_original[pd.notna(data_original[:, index_att]), :][:,-1] == pri_class)
    if index_att in categorical_features: 
        unique_values = np.unique(data_original[:,index_att][~pd.isna(data_original[:,index_att])])
    else:# 1 is prival, 0 is unpri_val after changing the continues values to the binary
        att_threshold = gain_ratio_numeric(data_original, index_att, weights)[2]
        data_original[pd.notna(data_original[:,index_att]), index_att] = np.where(data_original[pd.notna(data_original[:,index_att]), index_att] > att_threshold, 1, 0)
        unique_values = np.unique(data_original[:,index_att][~pd.isna(data_original[:,index_att])]).tolist()
    unique_list = []
    prob_list = []
    for u in unique_values:
        unique_list.append(u)
        try:
            filter_data = data_original[data_original[:,index_att] == u, :]
        except:
            filter_data = []
        try:
            result = sum(filter_data[:,-1] == pri_class)
        except:
            result = 0
        try:
            lenght_filter = len(filter_data[:,index_att])
        except:
            lenght_filter = 0
            
        if missing_positive == 0 or known_lenght_positive == 0:
            Fraction = 0
        else:
            Fraction = missing_positive * (result/known_lenght_positive)   
        if result == 0 or lenght_filter == 0:
            probability = 0
        else:
            probability = (result + Fraction) / (lenght_filter + number_of_missing)
        prob_list.append(probability)
    return sum(prob_list)
    

def Treatment_diffs(data, categorical_features, available_attributes, pro_index, pri_val, unpri_val, pri_class):
    DTs = []
    ind = []
    predictors = available_attributes
    for i in range(len(predictors)):
        if np.unique(data[:,i][~pd.isna(data[:,i])]).tolist() != []:
            if i != pro_index and predictors[i] == True:
                ind.append(i)
                DT = abs(DT_LEFT(data, pro_index, i, available_attributes, categorical_features, pri_val, unpri_val, pri_class) - DT_RIGHT(data, i, available_attributes, categorical_features, pri_class))
                DTs.append(DT)

            if i == pro_index and predictors[i] == True:
                ind.append(i)
                DT = -1
                DTs.append(DT)
        else:
            ind.append(i)
            DT = -1
            DTs.append(DT)
    return ind, DTs            
    
        
        





def search_best_attribute(data, available_attributes, categorical_features, weights):
    predictors = available_attributes
    if predictors.count(True) == 0:
            # no attributes left to choose
        return None, (0, 0, 0)

    candidates = []
    new = []
    average_gain = 0
    numerical_names = numerical_attributes(data, categorical_features)
    for i in range(len(predictors)):     
        if predictors[i] == True:
            if i in numerical_names:
                c = i, gain_ratio_numeric(data, i, weights)
            else:
                c = i, gain_ratio(data, i, weights)
            if c[1][1] > 0:
                average_gain += c[1][1]
                candidates.append(c)
    if len(candidates) == 0:
        # no suitable attribute
        return None, (0, 0, 0)
    average_gain = (average_gain / len(candidates)) - EPSILON
    # [0] gain ratio
    # [1] gain
    # [2] split informaton / threshold
    gain_ratios_table = []
    gain_table = []
    RANKS_GR = []
    selected = []
    best = (ZERO, ZERO, ZERO)
    for i, c in candidates:
        if c[1] >= average_gain:
            selected.append(i)
            gain_ratios_table.append(c[0])
            gain_table.append(c[1])
        else:
            selected.append(i)
            gain_ratios_table.append(0)
            gain_table.append(0)           
               
    return selected, gain_ratios_table





def numpy_table(data, pro_index, categorical_features, available_attributes, pri_val, unpri_val, pri_class, weights):
    if available_attributes.count(True) >= 1:
        g = search_best_attribute(data, available_attributes, categorical_features, weights)
        d = Treatment_diffs(data, categorical_features, available_attributes, pro_index, pri_val, unpri_val, pri_class)
        f = fairness_diffs(data, categorical_features, available_attributes, pro_index, pri_val, unpri_val, pri_class)
        np_table = np.zeros((len(g[1]), 6), dtype=float) 
        #columns = ['Index of attribute', 'DI', 'CV', 'Consistency', 'Gain Ratio', 'Dipsrate Treatment']
        if g[0] != None:# > 0:
            for i in range(len(g[0])):
                np_table[i, 0] = g[0][i]
                for j in range(len(f[0])):
                    #print(d[0], c[0])
                    if np_table[i,0] == f[0][j]:
                        np_table[i,1] = f[1][j]
                        np_table[i,2] = f[2][j]
                        np_table[i,3] = f[3][j]
                        np_table[i,4] = g[1][i]
                        np_table[i,5] = d[1][j]        
        else:
            np_table = np.array([])

    return np_table





def distance(np_table):
    for col in range(np.shape(np_table)[1]):
        if col == 1:
            np_table[:,col] = (rankdata(abs(np_table[:,col] - 1), method = "dense")).astype(int)    
        elif col == 2:
            np_table[:,col] = (rankdata(abs(np_table[:,col] - 0), method = "dense")).astype(int)
        elif col == 3:
            np_table[:,col] = (rankdata(abs(np_table[:,col] - 1), method = "dense")).astype(int)
        elif col == 4:
            np_table[:,col] = (rankdata(abs(np_table[:,col] - 1), method = "dense")).astype(int)
        elif col == 5:
            np_table[:,col] = (rankdata(abs(np_table[:,col] - 0), method = "dense")).astype(int)
    return np_table





def dominated_attribute(np_table):
    dominateds = []
    number_of_metrics = 5
    np_table1 = np_table 
    for i in range(np.shape(np_table1)[0]):
        for j in range(np.shape(np_table1)[0]):
            if sum((np_table[i,1:] < np_table1[j, 1:]) == True) == number_of_metrics:
                if sum((np_table[i,1:] == np_table1[j, 1:]) == True) != number_of_metrics:
                    dominateds.append(np_table1[j,0])
                    print("{} dominated {}".format(np_table[i, 0], np_table1[j,0]))
    lean_dominateds = []
    for d in dominateds:
        if d not in lean_dominateds:
            lean_dominateds.append(d)
    return lean_dominateds





def drop_dominateds(np_table):
    dominated_attributes = dominated_attribute(distance(np_table))
    for index in dominated_attributes:
        i = np.where(distance(np_table)[:,0] == index)
        np_table = np.delete(np_table, i, 0) #axis=0, delete a row that is index.an_array = np.delete(an_array, (1, 2), axis=0)
    return np_table





def lexicographical(np_table):
    order = ["GAIN_RATIOS", "DISPARATE_IMPACT", "CONSISTENCY", "CV_SCORE"]
    index_order = [4, 1, 3, 2, 5]

    min_index_4 = []    
    min_val_1 = []
    min_index_1 = []
    min_val_3 = []
    min_index_3 = []
    min_val_2 = []
    min_index_2 = []
    min_val_5 = []
    min_index_5 = []
    if sum(np_table[:,4] == min(np_table[:,4])) == 1:
        selected_index = np_table[np.argmin(np_table[:,4]),0]
    else:
        for j in range(len(np_table[:,4])):
            if np_table[j, 4] == min(np_table[:,4]):
                min_index_4.append(j)
        min_idx = None
        min_val = len(np_table) + 10
        for j in min_index_4:
            min_val_1.append((j, np_table[j,1]))
        for j, v in min_val_1:
            if v < min_val:
                min_val = v
                min_idx = j
            else:
                if v == min_val:
                    min_index_1.append(min_idx)
                    min_index_1.append(j)       
        if len(min_index_1) == 0:
            selected_index = np_table[min_idx, 0]
        else:
            min_idx = None
            min_val = len(np_table) + 10
            for k in min_index_1:
                min_val_3.append((k, np_table[k, 3]))
            for k, v in min_val_3:
                if v < min_val:
                    min_val = v
                    min_idx = k
                else:
                    if v == min_val:
                        min_index_3.append(min_idx)
                        min_index_3.append(k)
            if len(min_index_3) == 0:
                selected_index = np_table[min_idx, 0]
            else:
                min_idx = None
                min_val = len(np_table) + 10
                for l in min_index_3:
                    min_val_2.append((l, np_table[l, 2]))
                for l, v in min_val_2:
                    if v < min_val:
                        min_val = v
                        min_idx = l
                    else:
                        if v == min_val:
                            min_index_2.append(min_idx)
                            min_index_2.append(l)
                if len(min_index_2) == 0:
                    selected_index = np_table[min_idx,0]
                else:
                    min_idx = None
                    min_val = len(np_table) + 10
                    for l in min_index_2:
                        min_val_5.append((l, np_table[l, 5]))
                    for l, v in min_val_5:
                        if v < min_val:
                            min_val = v
                            min_idx = l
                        else:
                            if v == min_val:
                                min_index_5.append(min_idx)
                                min_index_5.append(l)
                    if len(min_index_5) == 0:
                        selected_index = np_table[min_idx,0]
                    else:
                        selected_index = None
    return selected_index





def search_lexi(np_table):    
    if len(np_table) == 0:
        lex_idx = None
    else:
        if len(drop_dominateds(np_table)) >= 2:
            new_table = drop_dominateds(np_table)
        else:
            new_table = distance(np_table)
        lex_idx = lexicographical(new_table)

        empty_list = []
        if lexicographical(new_table) == None:
            random_choose = new_table[np.random.randint(new_table.shape[0], size=1)[0], :]
            lex_idx = random_choose[0]
    return lex_idx





coefficient_value = [0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.40, 1.00]
deviation = [4.0, 3.09, 2.58, 2.33, 1.65, 1.28, 0.84, 0.25, 0.00]
CF = 0.25

i = 0

while CF > coefficient_value[i]:
    i += 1

coefficient = deviation[i - 1] + (deviation[i] - deviation[i - 1]) * (CF - coefficient_value[i - 1]) / (coefficient_value[i] - coefficient_value[i - 1])
coefficient = coefficient * coefficient

def estimate_error(total, errors):


    if total == 0:
        return 0
    elif errors < 1e-6:
        return total * (1 - math.exp(math.log(CF) / total))
    elif errors < 0.9999:
        v = total * (1 - math.exp(math.log(CF) / total))
        return v + errors * (estimate_error(total, 1.0) - v)
    elif errors + 0.5 >= total:
        return 0.67 * (total - errors)
    else:
        pr = (errors + 0.5 + coefficient / 2 + math.sqrt(coefficient * ((errors + 0.5) * (1 - (errors + 0.5) / total) + coefficient / 4))) / (total + coefficient)
        return (total * pr - errors)





def calculate_majority(class_attribute, weights):
    
    majority = []
    best = 0
    
    for value in np.unique(class_attribute): 
        count = weights[class_attribute == value].sum()
        
        if count > best:
            majority.clear()
            majority.append(value)
            best = count
        elif count == best:
            majority.append(value)
    
    return majority[random.randrange(len(majority))] if len(majority) > 0 else None





def pre_prune(metadata, data, majority, node, weights):

    class_attribute = data[:, -1]
    length = weights.sum()
    correct_predictions = 0

    if length > 0:
        majority = calculate_majority(class_attribute, weights)
        correct_predictions = weights[class_attribute == majority].sum()

    leaf_error = length - correct_predictions 
    
    if node.error() >= leaf_error - EPSILON:
        # class value : count
        distribution = Counter()

        for value in metadata[len(df.columns)-1]:
            distribution[int(value)] = weights[class_attribute == value].sum()

        return Node(majority, node.parent, leaf_error, length, distribution)
    
    return node





class Operator:
    """
    Enum-like class to represent different operators
    """
    
    EQUAL = 1
    LESS_OR_EQUAL = 2
    GREATER = 3

class Node:
    """
    A class used to represent a node of the decision tree.
    
    Each node can have a number of child nodes (internal nodes) or none (leaf nodes).
    The root of the tree is also represented as a node.
    """

    def __init__(self, attribute, parent=None, error=0, total=0, distribution=None, index=np.nan, available_attributes=np.nan):
        """
        Parameters
        ----------
        attribute : str
            The name of the attribute represented by the node (internal nodes) or
            the class value predicted (leaf nodes)
        parent : Node, optional
            The parent node of the node
        error:
            The number of prediction errors (leaf nodes)
        total:
            The number of instances reaching the node (leaf nodes)
        """
        
        
        self.attribute = attribute
        self.parent = parent

        # private-like attributes
        self._error = error 
        self._total = total
        self._distribution = distribution
        
        self.level = 0 if parent is None else parent.level + 1
        
        self.children = []
        self.conditions = []
        self.operators = []
        
        self.index = index 
        self.available_attributes =  available_attributes
        
        
    @property
    def classes(self):
        """Return the list of classes that the node (tree) can predict.
        
        This method can only be used on the root node of the tree.
        
        Returns
        -------
        list
            the list of classes that the node (tree) can predict.
        """

        return self._classes

    @classes.setter
    def classes(self, classes):
        """Set the list of classes that the node (tree) can predict.
        
        This list is used to determine the order of the classification probabilities.
        
        Parameters
        ----------
        classes : list
            The list of class values that the node can predict.
        """

        self._classes = classes

    def add(self, node, condition, operator=Operator.EQUAL):
        """Adds a child node
        
        The node will be added at the end of a branch. The condition and operator are
        used to decide when to follow the branch to the node
        
        Parameters
        ----------
        node : Node
            The node to add
        condition : str or float
            The value to decide when to choose to visit the node
        operator : Operator, optional
            The operator to decide when to choose to visit the node
        """
        
        node.parent = self
        self.children.append(node)
        self.conditions.append(condition)
        self.operators.append(operator)
   
    def to_text(self):
        """Prints a textual representation of the node
        
        This method prints the node and any of its child nodes
        """
        
        self.__print_node("")

    def __print_node(self, prefix):
        """Prints a textual representation of the node
        
        This method prints the node and any of its child nodes recusively
        
        Parameters
        ----------
        prefix : str
            The prefix to be used to print the node
        """

        if self.is_leaf():
            if self._error > 0:
                print("{} ({:.1f}/{:.1f})".format(self.attribute,
                                                  self._total,
                                                  self._error), end="")
            else:
                print("{} ({:.1f})".format(self.attribute,
                                           self._total), end="")
        else:
            if len(prefix) > 0:
                print("")

            for i, v in enumerate(self.conditions):
                if i > 0:
                    print("")

                operator = None

                if self.operators[i] == Operator.EQUAL:
                    operator = "="
                elif self.operators[i] == Operator.LESS_OR_EQUAL:
                    operator = "<="
                elif self.operators[i] == Operator.GREATER:
                    operator = ">"

                print("{}{} {} {}: ".format(prefix, self.attribute, operator, v), end="")

                self.children[i].__print_node(prefix +  "|    ")
    
    def is_leaf(self):
        """Checks whether the node is a leaf node
        
        Returns
        -------
        bool
            True if the node is a leaf node; otherwise False
        """
        
        return len(self.conditions) == 0
        
    def predict(self, instance):
        """Classify the specified instance
        
        This method expects that the instance (row) if a slice of a dataframe with the
        same attributes names as the one used to create the tree
        
        Parameters
        ----------
        instance : DataFrame slice
            The instance (row) to be classified
            
        Returns
        -------
        str
            The class value predicted
        
        """
        
        probabilities = Node.__predict(instance, self, 1.0)
        prediction = ("", 0)
        
        for value, count in probabilities.items():
            if count > prediction[1]:
                prediction = (value, count)
                
        if prediction[1] > 0:
            return prediction[0]
        
        raise Exception(f"Could not predict a value (probabilities '{probabilities}')")
        
    def probabilities(self, instance):
        """Classify the specified instance, returning the probability of each class value
        prediction.
        
        This method expects that the instance (row) is a slice of a dataframe with the
        same attributes names as the one used to create the tree. The order of the class
        values is determined by the ``self.classes`` property.
        
        Parameters
        ----------
        instance : DataFrame slice
            The instance (row) to be classified
            
        Returns
        -------
        list
            list of class value probabilities
        
        """
        
        probabilities = Node.__predict(instance, self, 1.0)
        prediction = []
        
        for value in self.classes:
            prediction.append(probabilities[value])
                
        return prediction

    def __predict(instance, node, weight):
        """Classify the specified instance
        
        This method expects that the instance (row) if a slice of a dataframe with the
        same attributes names as the one used to create the tree
        
        Parameters
        ----------
        instance : DataFrame slice
            The instance (row) to be classified
            
        Returns
        -------
        str
            The class value predicted
        
        """

        probabilities = Counter()
    
        # in case the node is a leaf
        if node.is_leaf():
            for value, count in node._distribution.items():
                probabilities[value] = weight * (count / node._total)

            if node._total == 0:
                probabilities[node.attribute] = weight 

            return probabilities
        
        # if not, find the branch to follow
        value = instance[node.index]
        
        if pd.isna(value):
            total = node.total()

            for i, v in enumerate(node.conditions):
                w = node.children[i].total() / total
                probabilities += Node.__predict(instance, node.children[i], weight * w)
        else:
            match = False

            for i, v in enumerate(node.conditions):
                if node.operators[i] == Operator.EQUAL and value == v:
                    match = True
                elif node.operators[i] == Operator.LESS_OR_EQUAL and value <= v:
                    match = True
                elif node.operators[i] == Operator.GREATER and value > v:
                    match = True
                    
                if match:
                    probabilities += Node.__predict(instance, node.children[i], weight)
                    break

            if not match:
                raise Exception(f"Cound not match value {value} for attribute {node.index}")
            
        return probabilities

    def total(self):
        """Returns the number of instances reaching the node
        
        For internal nodes, this is the sum of the total from its child nodes
        
        Returns
        -------
        int
            the number of instances reaching the node
        
        """

        if self.is_leaf():
            return self._total
        else:
            t = 0
            for node in self.children:
                t += node.total()
            return t
            
    def error(self):
        """Returns the number of prediction errors observed during the creation of the tree
        
        For internal nodes, this is the sum of the errors from its child nodes
        
        Returns
        -------
        int
            the number of prediction errors observed during the creation of the tree
        
        """

        if self.is_leaf():
            return self._error
        else:
            e = 0
            for node in self.children:
                e += node.error()
            return e
        
    def estimated(self):
        """Returns the number of estimated errors observed during the creation of the tree
        
        For internal nodes, this is the sum of the estimated errors from its child nodes
        
        Returns
        -------
        float
            the number of estimated errors observed during the creation of the tree
        
        """

        if self.is_leaf():
            return self._error + estimate_error(self._total, self._error)
        else:
            e = 0
            for node in self.children:
                e += node.estimated()
            return e
    
    def adjust(self, data):
        """Replaces the threshold values of continuous attributes with values that occur
        on the training data
        
        The discretisation uses the average value between two consecutive values to
        evaluate thresholds.
        
        Parameters
        ----------
        data : DataFrame
            The training data
        """

        if not self.is_leaf():
            ordered = []
            # we only need to look at one of the operators/conditions since the
            # threshold value will be the same in both branches
            operator = self.operators[0]
            threshold = self.conditions[0]

            if operator == Operator.LESS_OR_EQUAL or operator == Operator.GREATER:
                sorted_values = np.array(data[:, self.index])
                sorted_values.sort()
                selected = threshold
                
                for v in sorted_values:
                    if v > threshold:
                        break
                    selected = v
                
                self.conditions = [selected] * len(self.conditions)
                
            for child in self.children:
                child.adjust(data)
                
    def estimate_with_data(self, data, weights, update=False):
        """Returns the number of estimated errors observed on the specified data, updating
        the values if update=True (default False)
        
        For internal nodes, this is the sum of the estimated errors from its child nodes
        
        Parameters
        ----------
        data : DataFrame
            The data to use
        weights : np.array
            The instance weights to be used in the length calculation
        update : bool
            Indicate whether the error values should be updated or not
            
        Returns
        -------
        float
            the number of estimated errors
        """

        if self.is_leaf():
            class_attribute = data[:, -1]
            total = weights.sum()
            correct_predictions = 0 if total == 0 else weights[class_attribute == self.index].sum()
            error = total - correct_predictions
            
            if update:
                # class value = count
                distribution = Counter()

                for value in np.unique(class_attribute):
                    distribution[value] = weights[class_attribute == value].sum()
            
                self._distribution = distribution
                self._total = total
                self._error = error
                
            return error + estimate_error(total, error)
        else:
            missing = pd.isna(data[:,self.index])
            known_length = weights.sum() - weights[missing].sum()
            
            total = 0.0
            for i, v in enumerate(self.conditions):               
                if self.operators[i] == Operator.EQUAL:
                    partition = (data[:,self.index] == v) 
                elif self.operators[i] == Operator.LESS_OR_EQUAL:
                    partition = (data[:,self.index] <= v) 
                elif self.operators[i] == Operator.GREATER:
                    partition = (data[:,self.index] > v) 
                    
                updated_weights = weights.copy()
                w = weights[partition].sum() / known_length

                updated_weights[missing] = updated_weights[missing] * w
                updated_weights = updated_weights[partition | missing]

                if is_numeric_dtype(df.iloc[:,self.index]): 
                    branch_data = data[partition | missing]
                else:
                    branch_data = data[partition | missing]
                    

                total += self.children[i].estimate_with_data(branch_data, updated_weights, update)
            
            return total
        
    def sort(self):
        """Sort the branches of the node, placing leaf nodes at the start of the children
        array.
        
        This method improves the shape of the node (tree) for visualisation - there is
        no difference it terms of the performance of the tree.
        """

        for i in range(len(self.children)):
            if not self.children[i].is_leaf():
                to_index = -1
                
                for j in range(i + 1, len(self.children)):
                    if self.children[j].is_leaf():
                        to_index = j
                        break
                        
                if to_index == -1:
                    self.children[i].sort()
                else:
                    child = self.children[to_index]
                    condition = self.conditions[to_index]
                    operator = self.operators[to_index]
                    
                    for j in range(to_index, i, -1):
                        self.children[j] = self.children[j - 1]
                        self.conditions[j] = self.conditions[j - 1]
                        self.operators[j] = self.operators[j - 1]
                        
                    self.children[i] = child
                    self.conditions[i] = condition
                    self.operators[i] = operator


# In[ ]:


def post_prune(data, node, majority, weights):

    # (1) subtree error

    subtree_error = node.estimated()    

    # (2) leaf error
    
    class_attribute = data[:, -1]
    # class value = count
    distribution = Counter()
    leaf_total = 0

    for value in np.unique(class_attribute):
        distribution[value] = weights[class_attribute == value].sum()
        leaf_total += distribution[value]

    correct_predictions = 0 if leaf_total == 0 else distribution[majority]
    leaf_error = leaf_total - correct_predictions
    leaf_error += estimate_error(leaf_total, leaf_error)

    # (3) branch error

    selected = node.children[0]

    for i in range(1, len(node.children)):
        if selected.total() < node.children[i].total():
            selected = node.children[i]

    # checks whether to prune the node or not

    branch_error = float('inf')

    if selected.is_leaf():
        branch_error = leaf_error
    else:
        branch_error = selected.estimate_with_data(data, weights)

    if leaf_error <= (subtree_error + 0.1) and leaf_error <= (branch_error + 0.1):
        # replace by a leaf node
        return Node(majority, node.parent, leaf_error, leaf_total, distribution)
    elif branch_error <= (subtree_error + 0.1):
        # replace by the most common branch
        selected.estimate_with_data(data, weights, True)
        selected.parent = node.parent
        return selected
        
    return node





def build_decision_nodes(metadata, data, df, pro_index, categorical_features, available_attributes, pri_val, unpri_val, pri_class, tree=None, parent_majority=None, weights=None):

    if weights is None:
        # initializes the weights of the instances
        weights = np.ones(len(data))

    class_attribute = data[:, -1]
    is_unique = len(np.unique(class_attribute)) == 1
    length = weights.sum()

    # majority class (mode can return more than one value)
    majority = parent_majority if length == 0 else calculate_majority(class_attribute, weights)

    # if all instance belong to the same class or there is no enough data
    # a leaf node is added
    
    if is_unique or length < (MINIMUM_INSTANCES * 2):
        correct_predictions = 0 if length == 0 else weights[class_attribute == majority].sum()
        # class value = count
        distribution = Counter()

        for value in metadata[data.shape[1]-1]:
            distribution[int(value)] = weights[class_attribute == value].sum()

        return Node(majority, tree, length - correct_predictions, length, distribution)


    # search the best attribute for a split
    index = search_lexi(numpy_table(data, pro_index, categorical_features, available_attributes, pri_val, unpri_val, pri_class, weights))

    if index != None:        
        attribute = df.columns[int(index)]
    else:
        attribute = None
        
    
    if index==None:  
        # adds a leaf node, could not select an attribute
        correct_predictions = 0 if length == 0 else weights[class_attribute == majority].sum()
        # class value = count
        distribution = Counter()

        for value in metadata[len(df.columns)-1]:
            distribution[int(value)] = weights[class_attribute == value].sum()

        return Node(majority, tree, length - correct_predictions, length, distribution)

    # adjusts the instance weights based on missing values
    index = int(index)
    missing = pd.isna(data[:,index])
    known_length = length - weights[missing].sum()
    
    # (count, adjusted count)
    distribution = []
    
    if is_numeric_dtype(df.iloc[:,index]):
        #lower partition
        count = weights[data[:,index] <= gain_ratio_numeric(data, index, weights)[2]].sum()
        w = count / known_length
        adjusted_count = count + (weights[missing] * w).sum()
        distribution.append((count, adjusted_count))
    
        
        # upper partition
        count = weights[data[:,index] > gain_ratio_numeric(data, index, weights)[2]].sum()
        w = count / known_length
        adjusted_count = count + (weights[missing] * w).sum()
        distribution.append((count, adjusted_count))
    else:
        for value in  metadata[index]:
            count = weights[data[:,index] == value].sum()
            w = count / known_length
            adjusted_count = count + (weights[missing] * w).sum()
            distribution.append((count, adjusted_count))

    # only adds an internal node if there are enough instances for at least two branches
    valid = 0        
    for _, adjusted_count in distribution:
        if adjusted_count >= MINIMUM_INSTANCES:
            valid += 1
            
    node = None

    if valid < 2:
        reduced =  available_attributes.copy()#.clone()
        reduced[index] = False
        # not enough instances on branches, need to select another attribute

        return build_decision_nodes(metadata,
                                    data,
                                    df,
                                    pro_index,
                                    categorical_features,
                                    reduced,
                                    pri_val,
                                    unpri_val,
                                    pri_class,
                                    tree,
                                    parent_majority,
                                    weights)
    else:
        node = Node(attribute, parent=tree, index=index, available_attributes=available_attributes)

        if is_numeric_dtype(df.iloc[:,index]):
            # continuous threshold value
            threshold = gain_ratio_numeric(data, index, weights)[2]
            
            # slice if the data where the value <= threshold (lower)
            partition = data[:,index] <= threshold
            
            updated_weights = weights.copy()
            updated_weights[missing] = updated_weights[missing] * (distribution[0][0] / known_length)
            updated_weights = updated_weights[partition | missing]
            
            branch_data = data[partition | missing]
            child = build_decision_nodes(metadata, branch_data, df, pro_index, categorical_features, available_attributes, pri_val, unpri_val, pri_class, node, majority, updated_weights)
            node.add(pre_prune(metadata, branch_data, majority, child, updated_weights),
                     threshold,
                     Operator.LESS_OR_EQUAL)

            # slice if the data where the value > threshold (upper)
            partition = data[:, index] > threshold
            
            updated_weights = weights.copy()
            updated_weights[missing] = updated_weights[missing] * (distribution[1][0] / known_length)
            updated_weights = updated_weights[partition | missing]
            
            branch_data = data[partition | missing]
            child = build_decision_nodes(metadata, branch_data, df, pro_index, categorical_features, available_attributes, pri_val, unpri_val, pri_class, node, majority, updated_weights)
            node.add(pre_prune(metadata, branch_data, majority, child, updated_weights),
                     threshold,
                     Operator.GREATER) 
        else:
            # categorical split
           
            available_attributes[index] = False
            for idx, value in enumerate(metadata[index]):
                partition = data[:,index] == value
                updated_weights = weights.copy()

                updated_weights[missing] = updated_weights[missing] * (distribution[idx][0] / known_length)
                updated_weights = updated_weights[partition | missing]

               
                branch_data = data[partition | missing]
                child = build_decision_nodes(metadata, branch_data, df, pro_index, categorical_features, available_attributes.copy(), pri_val, unpri_val, pri_class, node, majority, updated_weights)
                node.add(pre_prune(metadata, branch_data, majority, child, updated_weights), value, Operator.EQUAL)
        
        # checks whether to prune the node or not
        if PRUNE:
            node = post_prune(data, node, majority, weights)
        
    # if we are the root node of the tree
    if tree is None:
        node.adjust(data)
        node.sort()
        node.classes = np.unique(class_attribute)

    return node





def split_after_training(test, pro_index, pri_val):
    y_pred_val = list(test[test[:,pro_index] == pri_val, :][:,-1])
    y_val = list(test[test[:,pro_index] == pri_val, :][:,-2])
    return y_val, y_pred_val

def AUC(y, y_pred, pri_class, non_pri_class):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(y_pred)): 
        if y[i]==y_pred[i]== pri_class:
            TP += 1
        if y_pred[i]== pri_class and y[i]!=y_pred[i]:
            FP += 1
        if y[i]==y_pred[i]== non_pri_class:
            TN += 1
        if y_pred[i]== non_pri_class and y[i]!=y_pred[i]:
            FN += 1         
    try:
        fpr = FP/(FP+TN)
    except ZeroDivisionError:
        fpr = 0 
    try:
        tpr = TP/(TP+FN)
    except ZeroDivisionError:
        tpr = 0 
    return 1/2 - fpr/2 + tpr/2

def FPR_rate(test, pro_index, pri_val, pri_class, non_pri_class): 
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    y, y_pred = split_after_training(test, pro_index, pri_val)
    for i in range(len(y_pred)): 
        if y[i]==y_pred[i]== pri_class:
            TP += 1
        if y_pred[i]== pri_class and y[i]!=y_pred[i]:
            FP += 1
        if y[i]==y_pred[i]== non_pri_class:
            TN += 1
        if y_pred[i]== non_pri_class and y[i]!=y_pred[i]:
            FN += 1
    try:
        return FP/(FP+TN)
    except ZeroDivisionError:

        return 0

def FNR_rate(test, pro_index, pri_val, pri_class, non_pri_class):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    y, y_pred = split_after_training(test, pro_index, pri_val)
    for i in range(len(y_pred)): 
        if y[i]==y_pred[i]== pri_class:
            TP += 1
        if y_pred[i]== pri_class and y[i]!=y_pred[i]:
            FP += 1
        if y[i]==y_pred[i]== non_pri_class:
            TN += 1
        if y_pred[i]== non_pri_class and y[i]!=y_pred[i]:
            FN += 1
    try:
        return FN/(FN+TP)
    except ZeroDivisionError:
        return 0
    

def DI_CV_Calculate(data, test, pro_index, categorical_features, pri_val, unpri_val, pri_class, user_defined_threshold=None):
    test_original = test.copy()
    weights = np.ones(len(test_original))
    if sum(pd.isna(test_original[:,pro_index])) > 0:
        lenght = weights.sum()
        number_of_missing =  weights[pd.isna(test_original[:,pro_index])].sum()
        missing_positive = sum(test_original[pd.isna(test_original[:, pro_index]), :][:,-1] == pri_class)
        known_length = lenght - number_of_missing
        known_lenght_positive = sum(test_original[pd.notna(test_original[:, pro_index]), :][:,-1] == pri_class)
    else:  
        lenght = weights.sum()
        number_of_missing = 0
        missing_positive = 0
        known_length = lenght
        known_lenght_positive = sum(test_original[pd.notna(test_original[:, pro_index]), :][:,-1] == pri_class)
        
    if pro_index in categorical_features: #transform in binary case if the index in categorical list
        unique_values = np.unique(data[:,pro_index][~pd.isna(data[:,pro_index])])
        test_original[(pd.notna(test_original[:,pro_index]) & (test_original[:,pro_index] != pri_val)), pro_index] = unpri_val
    else:# 1 is prival, 0 is unpri_val after changing the continues values to the binary
        weights_data =  np.ones(len(data))
        att_threshold = user_defined_threshold
        test_original[pd.notna(test_original[:,pro_index]), pro_index] = np.where(test_original[pd.notna(test_original[:,pro_index]), pro_index] > att_threshold, pri_val, unpri_val)
    unique_values = np.unique(test_original[:,pro_index][~pd.isna(test_original[:,pro_index])]).tolist()
    try:
        pri_data = test_original[test_original[:,pro_index] == pri_val, :] 
    except:
        pri_data = []
    try:
        unpri_data = test_original[test_original[:,pro_index] == unpri_val, :]
    except:
        unpri_data = []

    #number of priviliged values that assigned to positive class 
    #so, I used pri_data to reach priviliged values that are assigned to positive class
    try:
        privileged = sum(pri_data[:,-1] == pri_class) 
    except:
        privileged = 0
    try:
        lenght_pri = len(pri_data[:,pro_index])  
    except:
        lenght_pri = 0
    if known_lenght_positive == 0 or missing_positive == 0:
        F_privileged = 0
    else:
        F_privileged = missing_positive * privileged/known_lenght_positive
    
    if (privileged + F_privileged) == 0 or (lenght_pri + number_of_missing) == 0:
        privileged_probability = 0#try:
    else:
        privileged_probability = (privileged + F_privileged) / (lenght_pri + number_of_missing)
  
    try:
        unprivileged = sum(unpri_data[:,-1] == pri_class)
    except:
        unprivileged = 0
    try:
        lenght_unpri = len(unpri_data[:,pro_index])
    except:
        lenght_unpri = 0 
    if known_lenght_positive == 0 or missing_positive == 0:
        F_unprivileged = 0
    else:
        F_unprivileged = missing_positive * unprivileged/known_lenght_positive
    if (unprivileged + F_unprivileged) == 0 or (lenght_unpri + number_of_missing) == 0:
        unprivileged_probability = 0
    else:
        unprivileged_probability = (unprivileged + F_unprivileged) / (lenght_unpri + number_of_missing)

    if unprivileged_probability == 0 or privileged_probability == 0:
        DI_DATA = 0
    else:
        DI_DATA = unprivileged_probability / privileged_probability

        
    try:    
        CV_DATA = privileged_probability - unprivileged_probability
    except:
        CV_DATA = 0
        
    return pri_data, unpri_data, DI_DATA, CV_DATA 



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
def consistency_pre(data_split, categorical_features, y_split, pro_index, k=5):
    if len(data_split) < k:
        k = len(data_split)
    cols = []
    if len(data_split) > 0:
        [cols.append(i) for i in categorical_features if i!=pro_index]
        #ct = ColumnTransformer([("encoder", OneHotEncoder(), cols)], remainder = 'passthrough')
        #data_split = ct.fit_transform(data_split)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data_split)
        indices = nbrs.kneighbors(data_split, return_distance=False)
        return 1 - abs(y_split - y_split[indices].mean(axis=1)).mean()
    else:
        data_split = 0
        y_split = 0
        return 0
def consistency_test(data, test, pro_index, categorical_features, pri_val, unpri_val, pri_class,user_defined_threshold=None):
    weights = np.ones(len(test))
    try:
        pri_data = DI_CV_Calculate(data, test, pro_index, categorical_features, pri_val, unpri_val, pri_class,user_defined_threshold=None)[0]
        X_priviliged = pri_data[~np.isnan(pri_data).any(axis=1)][:,:-1]
        y_priviliged = pri_data[~np.isnan(pri_data).any(axis=1)][:,-1]
    except:
        pri_data = []
        X_priviliged = []
        y_priviliged = []
    consistency_pri = consistency_pre(X_priviliged, categorical_features, y_priviliged, pro_index, k=5)
    try:
        unpri_data = DI_CV_Calculate(data, test, pro_index, categorical_features, pri_val, unpri_val, pri_class,user_defined_threshold=None)[1]
        X_unpriviliged = unpri_data[~np.isnan(unpri_data).any(axis=1)][:,:-1]
        y_unpriviliged = unpri_data[~np.isnan(unpri_data).any(axis=1)][:,-1]
    except:
        unpri_data = []
        X_unpriviliged = []
        y_unpriviliged = []
    consistency_unpri = consistency_pre(X_unpriviliged, categorical_features, y_unpriviliged, pro_index, k=5)
    return consistency_pri + consistency_unpri





def DT_LEFT_TEST(data, test, pro_index, index_att, available_attributes, categorical_features, pri_val, unpri_val, pri_class, user_defined_threshold=None):
    weights = np.ones(len(data))
    PRI = DI_CV_Calculate(data, test, pro_index, categorical_features, pri_val, unpri_val, pri_class, user_defined_threshold=None)[0]
    UNPRI = DI_CV_Calculate(data, test, pro_index, categorical_features, pri_val, unpri_val, pri_class, user_defined_threshold=None)[1]
    weights_pri = np.ones(len(PRI))
    weights_unpri = np.ones(len(UNPRI))
    test_original = test.copy()
    data_original = data
    if sum(pd.isna(test_original[:,index_att])) > 0:
        number_missing_pri = weights_pri[pd.isna(PRI[:,index_att])].sum()
        known_lenght_pri = weights_pri.sum() - number_missing_pri
        missing_positive_pri = sum(PRI[pd.isna(PRI[:, index_att]), :][:,-1] == pri_class)
        known_lenght_positive_pri = sum(PRI[pd.notna(PRI[:, index_att]), :][:,-1] == pri_class)
        
        number_missing_unpri = weights_unpri[pd.isna(UNPRI[:,index_att])].sum()
        known_lenght_unpri = weights_unpri.sum() - number_missing_unpri
        missing_positive_unpri = sum(UNPRI[pd.isna(UNPRI[:, index_att]), :][:,-1] == pri_class)
        known_lenght_positive_unpri = sum(UNPRI[pd.notna(UNPRI[:, index_att]), :][:,-1] == pri_class)
    else:
        number_missing_pri = 0
        known_lenght_pri = weights_pri.sum() - number_missing_pri
        missing_positive_pri = 0
        known_lenght_positive_pri = sum(PRI[pd.notna(PRI[:, index_att]), :][:,-1] == pri_class)
        
        number_missing_unpri = 0
        known_lenght_unpri = weights_unpri.sum()
        missing_positive_unpri = 0
        known_lenght_positive_unpri = sum(UNPRI[pd.notna(UNPRI[:, index_att]), :][:,-1] == pri_class)
    if index_att in categorical_features:
        unique_values = np.unique(test_original[:,index_att][~pd.isna(test_original[:,index_att])])
    else:
        att_threshold = gain_ratio_numeric(data_original, index_att, weights)[2]
        test_original[pd.notna(test_original[:,index_att]), index_att] = np.where(test_original[pd.notna(test_original[:,index_att]), index_att] > att_threshold, 1, 0)
        PRI[pd.notna(PRI[:,index_att]), index_att] = np.where(PRI[pd.notna(PRI[:,index_att]), index_att] > att_threshold, 1, 0)
        UNPRI[pd.notna(UNPRI[:,index_att]), index_att] = np.where(UNPRI[pd.notna(UNPRI[:,index_att]), index_att] > att_threshold, 1, 0)  
        unique_values = np.unique(test_original[:,index_att][~pd.isna(test_original[:,index_att])]).tolist()  
    unique_list = []
    probs_privals = []
    probs_unprivals = []
    for u in unique_values:
        unique_list.append(u)
        try:
            filter_data_priviliged = PRI[PRI[:,index_att] == u, :]
        except:
            filter_data_priviliged = []
        try:
            filter_data_unpriviliged = UNPRI[UNPRI[:,index_att] == u, :]
        except:
            filter_data_unpriviliged = []
        try:
            result_pri = sum(filter_data_priviliged[:,-1] == pri_class)
        except:
            result_pri = 0
        try:
            lenght_pri = len(filter_data_priviliged[:,index_att])
        except:
            lenght_pri = 0
        if missing_positive_pri == 0 or known_lenght_positive_pri == 0:
            F_pri = 0
        else:
            F_pri = missing_positive_pri * (result_pri/known_lenght_positive_pri)   
        if result_pri == 0 or lenght_pri == 0:
            prob_pri = 0
        else:
            prob_pri = (result_pri + F_pri) / (lenght_pri + number_missing_pri)
        probs_privals.append(prob_pri)
        
        
        try:
            result_unpri = sum(filter_data_unpriviliged[:,-1] == pri_class)
        except:
            result_unpri = 0
        try:
            lenght_unpri = len(filter_data_unpriviliged[:,index_att])
        except:
            lenght_unpri = 0
        if missing_positive_unpri == 0 or known_lenght_positive_unpri == 0:
            F_unpri = 0
        else:
            F_unpri = missing_positive_unpri * (result_unpri/known_lenght_positive_unpri)   
        if result_unpri == 0 or lenght_unpri == 0:
            prob_unpri = 0
        else:
            prob_unpri = (result_unpri + F_unpri) / (lenght_unpri + number_missing_unpri)
        probs_unprivals.append(prob_unpri)
        
    return sum(probs_privals) + sum(probs_unprivals)
            
            
def DT_RIGHT_RIGHT(data, test, index_att, available_attributes, categorical_features, pri_class):
    test_original = test.copy()
    data_original = data.copy()
    weights = np.ones(len(data))
    weights_test = np.ones(len(test))
    if sum(pd.isna(test_original[:,index_att])) > 0:
        lenght = weights_test.sum()
        number_of_missing = weights_test[pd.isna(test_original[:,index_att])].sum()
        missing_positive = sum(test_original[pd.isna(test_original[:, index_att]), :][:,-1] == pri_class)
        known_length = lenght - number_of_missing
        known_lenght_positive = sum(test_original[pd.notna(test_original[:, index_att]), :][:,-1] == pri_class)
    else: 
        lenght = weights_test.sum()
        number_of_missing = 0
        missing_positive = 0
        known_length = lenght
        known_lenght_positive = sum(test_original[pd.notna(test_original[:, index_att]), :][:,-1] == pri_class)
    if index_att in categorical_features: 
        unique_values = np.unique(test_original[:,index_att][~pd.isna(test_original[:,index_att])])
    else:# 1 is prival, 0 is unpri_val after changing the continues values to the binary
        att_threshold = gain_ratio_numeric(data_original, index_att, weights)[2]
        test_original[pd.notna(test_original[:,index_att]), index_att] = np.where(test_original[pd.notna(test_original[:,index_att]), index_att] > att_threshold, 1, 0)
        unique_values = np.unique(test_original[:,index_att][~pd.isna(test_original[:,index_att])]).tolist()
    unique_list = []
    prob_list = []
    for u in unique_values:
        unique_list.append(u)
        try:
            filter_data = test_original[test_original[:,index_att] == u, :]
        except:
            filter_data = []
        try:
            result = sum(filter_data[:,-1] == pri_class)
        except:
            result = 0
        try:
            lenght_filter = len(filter_data[:,index_att])
        except:
            lenght_filter = 0
            
        if missing_positive == 0 or known_lenght_positive == 0:
            Fraction = 0
        else:
            Fraction = missing_positive * (result/known_lenght_positive)   
        if result == 0 or lenght_filter == 0:
            probability = 0
        else:
            probability = (result + Fraction) / (lenght_filter + number_of_missing)
        prob_list.append(probability)
    return sum(prob_list)
    

def Treatment_test_diffs(data, test, categorical_features, available_attributes, pro_index, pri_val, unpri_val, pri_class,user_defined_threshold=None):

    DTs = []
    ind = []
    for i in range(len(test[0,:-2])):
        if np.unique(test[:,i][~pd.isna(test[:,i])]).tolist() != []:
            if i != pro_index:
                ind.append(i)
                DT = abs(DT_LEFT_TEST(data, test, pro_index, i, available_attributes, categorical_features, pri_val, unpri_val, pri_class, user_defined_threshold=None) - DT_RIGHT_RIGHT(data, test, i, available_attributes, categorical_features, pri_class))
                DTs.append(DT)

                
        else:
            ind.append(i)
            DT = -1
            DTs.append(DT)
    return ind, DTs      

    



def run(metadata, data, df, data_test, pro_index, categorical_features, pri_val, unpri_val, pri_class, non_pri_class):
    test = data_test
    train = data
    available_attributes = available_attribute(train)
    tree = build_decision_nodes(metadata, train, df, pro_index, categorical_features, available_attributes, pri_val, unpri_val, pri_class, tree=None, parent_majority=None, weights=None)
    
    print("\n\n===> Printing Tree\n")
    tree.to_text()
    y_pred = []
    correct = 0
    for i in range(len(test)):
        prediction = tree.predict(test[i,:-1])
        y_pred.append(prediction)

        if prediction == test[i, -1]:
            correct += 1
    acc =  correct / len(test)

    y = test[:,-1]
    col = np.array(y_pred)
    test = np.column_stack((test,col))
    roc_auc= AUC(y, y_pred, pri_class, non_pri_class)

    DI = DI_CV_Calculate(data, test, pro_index, categorical_features, pri_val, unpri_val, pri_class,user_defined_threshold=None)[2] 
    CV = DI_CV_Calculate(data, test, pro_index, categorical_features, pri_val, unpri_val, pri_class,user_defined_threshold=None)[3] 
    DT = Treatment_test_diffs(data, test, categorical_features, available_attributes, pro_index, pri_val, unpri_val, pri_class,user_defined_threshold=None)[1]
    cons= consistency_test(data, test, pro_index, categorical_features, pri_val, unpri_val, pri_class,user_defined_threshold=None)
    weights =  np.ones(len(data)) 
    test_original = test
    if pro_index in categorical_features:
        unique_values_protected = np.unique(data[:,pro_index][~pd.isna(data[:,pro_index])])
        if len(unique_values_protected) > 2:
            test_original[(pd.notna(test_original[:,pro_index]) & (test_original[:,pro_index] != pri_val)), pro_index] = unpri_val
    else:

        att_threshold = gain_ratio_numeric(data, pro_index, weights)[2]
        test_original[pd.notna(test_original[:,pro_index]), pro_index] = np.where(test_original[pd.notna(test_original[:,pro_index]), pro_index] > att_threshold, pri_val, unpri_val)

    FPR_UND = FPR_rate(test_original, pro_index, pri_val, pri_class, non_pri_class)
    FPR_DIS = FPR_rate(test_original, pro_index, unpri_val, pri_class, non_pri_class)
    FPR_Diff=abs(FPR_UND - FPR_DIS) 

    FNR_UND = FNR_rate(test_original, pro_index, pri_val, pri_class, non_pri_class)
    FNR_DIS = FNR_rate(test_original, pro_index, unpri_val, pri_class, non_pri_class)
    FNR_Diff=abs(FNR_UND - FNR_DIS)

    print('\n\n===>ACCURACY:', acc)
    print('\n\n===>ROC Result:', roc_auc)
    print('\n\n===>FPR_diffs Result: ', FPR_Diff)
    print('\n\n===>FNR_diffs Result: ', FNR_Diff)
    print('\n\n===>Disparate Impact Result: ', DI)
    print('\n\n===>CV score Result: ', CV)
    print('\n\n===>Consistency Result: ', cons)
    print('\n\n===>Disparate Treatment Result: ', np.mean(DT))
    return ('ACCURACY:',acc), ('ROC_AUC:', roc_auc), ('FPR RESULT:', FPR_Diff), ('FNR RESULT:', FNR_Diff), ('DISPARATE IMPACT:', DI), ('CV SCORE:', CV), ('CONSISTENCY:', cons), ('DISPARATE TREATMENTS:', np.mean(DT))

run(metadata, data, df, data_test, 9,categorical_features, 1, 2, 1, 0)

