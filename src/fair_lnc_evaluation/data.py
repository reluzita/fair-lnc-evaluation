from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import openml
import pandas as pd
import numpy as np
import os

class Dataset:
    def __init__(self, data, name, sensitive_attr, target, test_size):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            The dataset.
        name : str
            The name of the dataset.
        sensitive_attr : str
            Feature to consider as the sensitive attribute.
        target : str
            The target variable.
        test_size : float
            The proportion of the dataset to include in the test split.
        """
        self.data = data
        self.name = name
        self.sensitive_attr = sensitive_attr
        self.target = target
        self.test_size = test_size

        self.X = self.data.drop(columns=[self.target])
        self.y = self.data[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=0, stratify=self.y)

    def get_labels(self, set=None):
        """
        Returns the labels of the dataset.

        Parameters
        ----------
        set : str, optional
            The set to return the labels for (train/test). If None, the labels for the whole dataset are returned.

        Returns
        -------
        pandas.Series
            The labels.
        """
        if set == 'train':
            return self.y_train
        elif set == 'test':
            return self.y_test
        elif set == None:
            return self.y
        else:
            raise ValueError('Invalid set.')
    
    def get_sensitive_attr(self, set=None):
        """
        Returns the sensitive attribute column of the dataset.

        Parameters
        ----------
        set : str, optional
            The set to return the sensitive attribute for (train/test). If None, the sensitive attribute for the whole dataset is returned.

        Returns
        -------
        pandas.Series
            The sensitive attribute column.
        """
        if set == 'train':
            return self.X_train[self.sensitive_attr]
        elif set == 'test':
            return self.X_test[self.sensitive_attr]
        elif set == None:
            return self.X[self.sensitive_attr]
        else:
            raise ValueError('Invalid set.')
        
    def get_features(self, set=None):
        """
        Returns the features of the dataset.

        Parameters
        ----------
        set : str, optional
            The set to return the features for (train/test). If None, the features for the whole dataset are returned.

        Returns
        -------
        pandas.DataFrame
            The features.
        """
        if set == 'train':
            return self.X_train
        elif set == 'test':
            return self.X_test
        elif set == None:
            return self.X
        else:
            raise ValueError('Invalid set.')
        
def format_adult():
    data, _, _, _ = openml.datasets.get_dataset(43898).get_data(dataset_format="dataframe")
    data = data.dropna().reset_index(drop=True)    

    data['y'] = data['class'].apply(lambda x: 1 if x == '>50K' else 0).astype('int')
    data = data.drop('class', axis=1)
    data = pd.get_dummies(data)
    data = data.drop(['native_country_?', 'workclass_?', 'occupation_?', 'sex_Female'], axis=1)

    return data

def format_german():
    data, _, _, _ = openml.datasets.get_dataset(31).get_data(dataset_format="dataframe")

    data['sex_Male'] = data['personal_status'].apply(lambda x: 1 if (x == 'male single' or x == 'male mar/wid' or x == 'male div/sep') else 0)
    data['single'] = data['personal_status'].apply(lambda x: 1 if (x == 'female single' or x == 'male single') else 0)
    data['own_telephone'] = data['own_telephone'].apply(lambda x: 1 if x == 'yes' else 0).astype('int')
    data['foreign_worker'] = data['foreign_worker'].apply(lambda x: 1 if x == 'yes' else 0).astype('int')
    data['y'] = data['class'].apply(lambda x: 1 if x == 'good' else 0).astype('int')
    data = data.drop(['personal_status','class'], axis=1)
    data = pd.get_dummies(data)

    return data

def format_compas():
    data, _, _, _ = openml.datasets.get_dataset(45039).get_data(dataset_format="dataframe")

    data['y'] = data['twoyearrecid'].astype('int')
    data = data.drop('twoyearrecid', axis=1)

    return data

def format_ricci():
    data, _, _, _ = openml.datasets.get_dataset(42665).get_data(dataset_format="dataframe")

    data['Position_Captain'] = data['Position'].apply(lambda x: 1 if x == 'Captain' else 0).astype('int')
    data['y'] = data['Promotion'].apply(lambda x: 1 if x == 'Promotion' else 0).astype('int')
    data = data.drop(['Promotion', 'Position'], axis=1)
    data = pd.get_dummies(data)

    return data

def format_diabetes():
    data, _, _, _ = openml.datasets.get_dataset(43903).get_data(dataset_format="dataframe")

    data['y'] = data['readmit_30_days'].astype('int')
    data = data.drop('readmit_30_days', axis=1)
    for col in ['medicare', 'medicaid', 'had_emergency', 'had_inpatient_days', 'had_outpatient_days']:
        data[col] = data[col].astype('int')
    data['change'] = data['change'].apply(lambda x: 1 if x == 'Ch' else 0).astype('int')
    data['diabetesMed'] = data['diabetesMed'].apply(lambda x: 1 if x == 'Yes' else 0).astype('int')
    data = pd.get_dummies(data)
    data = data.drop(['gender_Female', 'gender_Unknown/Invalid'], axis=1)

    return data

def format_phishing():
    data, _, _, _ = openml.datasets.get_dataset(4534).get_data(dataset_format="dataframe")

    data = data.astype(int)
    for col in data.columns:
        if data[col].value_counts().shape[0] == 2:
            data[col] = data[col].apply(lambda x: 1 if x == 1 else 0)

    data['y'] = data['Result']
    data = data.drop('Result', axis=1)

    return data

def format_titanic():
    data, _, _, _ = openml.datasets.get_dataset(40945).get_data(dataset_format="dataframe")

    data = data.drop(['cabin', 'ticket', 'boat', 'body', 'home.dest'], axis=1)
    data['sex'] = data['sex'].apply(lambda x: 1 if x == 'male' else 0).astype('int')
    data['title'] = data['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    data = data.drop(columns='name')
    data['title'] = data['title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
    data['title'] = data['title'].map({'Ms': 'Miss', 'Mme': 'Mrs', 'Mlle': 'Miss'})
    data['embarked'] = data['embarked'].fillna('S')
    NaN_indexes = data['age'][data['age'].isnull()].index
    for i in NaN_indexes:
        pred_age = data['age'][((data.sibsp == data.iloc[i]["sibsp"]) & (data.parch == data.iloc[i]["parch"]) & (data.pclass == data.iloc[i]["pclass"]))].median()
        if not np.isnan(pred_age):
            data['age'].iloc[i] = pred_age
        else:
            data['age'].iloc[i] = data['age'].median()
    for i, row in data.loc[data['fare'].isna()].iterrows():
        data.loc[i, 'fare'] = data.loc[data['pclass'] == data.loc[i, 'pclass']].fare.median()
    data['y'] = data['survived'].astype('int')
    data = data.drop('survived', axis=1)

    data = pd.get_dummies(data)

    return data

def format_bank():
    data, _, _, _ = openml.datasets.get_dataset(1461).get_data(dataset_format="dataframe")

    data = data.rename(columns={
        "V1": "age","V2": "job","V3": "marital","V4": "education","V5": "default","V6": "balance","V7": "housing","V8": "loan","V9": "contact",
        "V10": "day","V11": "month","V12": "duration","V13": "campaign","V14": "pdays","V15": "previous","V16": "poutcome", "V17": "y"
    })

    data['education'] = data['education'].map({'unknown': np.nan, 'primary':1, 'secondary':2, 'tertiary':3}).astype(float)
    data = data.dropna()

    for c in ['default', 'housing', 'loan']:
        data[c] = data[c].map({'no': 0, 'yes': 1}).astype(int)

    data['month'] = data['month'].map({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}).astype(int)
    data['y'] = data['Class'].astype(int).apply(lambda x: 1 if x == 2 else 0).astype(int)
    data = data.drop(columns=['Class'])

    data = pd.get_dummies(data)
    data = data.drop(columns=['job_unknown', 'marital_single', 'contact_unknown', 'poutcome_unknown'])

    return data

def format_monks(n):
    if n == 1:
        data, _, _, _ = openml.datasets.get_dataset(333).get_data(dataset_format="dataframe")
    elif n == 2:
        data, _, _, _ = openml.datasets.get_dataset(334).get_data(dataset_format="dataframe")
    data = data.astype(int)

    data[['attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6']] = data[['attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6']].apply(lambda x: x-1)
    data = data.rename(columns={'class': 'y'})

    return data

def format_biodeg():
    data, _, _, _ = openml.datasets.get_dataset(1494).get_data(dataset_format="dataframe")

    data['y'] = data['Class'].map({'1':0, '2':1}).astype(int)
    data = data.drop(columns=['Class'])

    return data

def format_credit():
    data, _, _, _ = openml.datasets.get_dataset(29).get_data(dataset_format="dataframe")
    data = data.dropna()

    data['A1'] = data['A1'].map({'a':1, 'b':0}).astype(int)
    for col in ['A9' ,'A10', 'A12']:
        data[col] = data[col].map({'t':1, 'f':0}).astype(int)
    data['y'] = data['class'].map({'-':0, '+':1}).astype(int)
    data = data.drop(columns=['class'])

    data = pd.get_dummies(data)

    return data

def format_sick():
    data, _, _, _ = openml.datasets.get_dataset(38).get_data(dataset_format="dataframe")

    data = data.drop(columns=['TBG', 'TBG_measured', 'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured'])
    data = data.dropna()
    data['sex'] = data['sex'].map({'F':0, 'M':1}).astype(int)
    for col in ['on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych']:
        data[col] = data[col].map({'f':0, 't':1}).astype(int)
    data['y'] = data['Class'].map({'negative':0, 'sick':1}).astype(int)
    data = data.drop(columns='Class')
    data = pd.get_dummies(data)

    return data

def format_churn():
    data, _, _, _ = openml.datasets.get_dataset(40701).get_data(dataset_format="dataframe")
    for col in ['international_plan', 'voice_mail_plan', 'number_customer_service_calls']:
        data[col] = data[col].astype(int)
    data['y'] = data['class'].astype(int)
    data = data.drop(columns='class')
    data = pd.get_dummies(data)

    return data

def format_vote():
    data, _, _, _ = openml.datasets.get_dataset(56).get_data(dataset_format="dataframe")
    data = data.drop(columns=['water-project-cost-sharing', 'export-administration-act-south-africa'])
    data = data.dropna()
    for col in data.columns:
        if col != 'Class':
            data[col] = data[col].map({'n':0, 'y':1}).astype(int)

    data['y'] = data['Class'].map({'republican':0, 'democrat':1}).astype(int)
    data = data.drop(columns=['Class'])

    return data

def format_ads():
    data, _, _, _ = openml.datasets.get_dataset(40978).get_data(dataset_format="dataframe")
    data['y'] = data['class'].map({'noad':0, 'ad':1}).astype(int)
    data = data.drop(columns=['class'])
    data = data.astype(int)

    return data

def format_soil():
    data, _, _, _ = openml.datasets.get_dataset(923).get_data(dataset_format="dataframe")
    data['isns'] = data['isns'].astype(int)
    data['y'] = data['binaryClass'].map({'N':0, 'P':1}).astype(int)
    data = data.drop(columns=['binaryClass'])

    return data


def get_data(name, sensitive_attr, test_size=0.2, save=False):
    """
    Get dataset from OpenML, format it for the experiments and return a Dataset object

    Parameters
    ----------
    name : str
        Name of the dataset
    sensitive_attr : str
        Name of the sensitive attribute
    test_size : float, optional
        Size of the test set
    save : bool, optional
        Whether to save the dataset as a csv file
    """
    if save and os.path.exists(f'datasets/{name}.csv'):
        data = pd.read_csv(f'datasets/{name}.csv')

        if sensitive_attr not in data.columns:
            raise ValueError(f'{sensitive_attr} is not a column of {name}')
        
        return Dataset(data, name, sensitive_attr, 'y', test_size)
    
    if name == 'adult':
        data = format_adult() 
    elif name == 'german':
        data = format_german()
    elif name == 'compas':
        data = format_compas()
    elif name == 'ricci':
        data = format_ricci()
    elif name == 'diabetes':
        data = format_diabetes()
    elif name == 'phishing':
        data = format_phishing()
    elif name == 'titanic':
        data = format_titanic()
    elif name == 'bank':
        data = format_bank()
    elif name == 'monks1':
        data = format_monks(1)
    elif name == 'monks2':
        data = format_monks(2)
    elif name == 'biodeg':
        data = format_biodeg()
    elif name == 'credit':
        data = format_credit()
    elif name == 'sick':
        data = format_sick()
    elif name == 'churn':
        data = format_churn()
    elif name == 'vote':
        data = format_vote()
    elif name == 'ads':
        data = format_ads()
    elif name == 'soil':
        data = format_soil()

    X = data.drop('y', axis=1)
    y = data['y']

    if y.value_counts()[0]/len(y) < 0.2 or y.value_counts()[1]/len(y) < 0.2:
        print('Dataset is too unbalanced, performing random undersampling')
        X, y = RandomUnderSampler(sampling_strategy=0.5, random_state=42).fit_resample(X, y)
        data = X.copy()
        data['y'] = y

    if save:
        if not os.path.exists('datasets'):
            os.mkdir('datasets')
        data.to_csv(f'datasets/{name}.csv', index=False)

    if sensitive_attr not in data.columns:
        raise ValueError(f'{sensitive_attr} is not a column of {name}')

    return Dataset(data, name, sensitive_attr, 'y', test_size)