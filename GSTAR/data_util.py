import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler, MaxAbsScaler
import os, urllib
from .helper import *
from .plot import *
from .fairness import *

def get_dataset(name, save=False, corr_sens=False, seed=42, verbose=False):
    """
    Retrieve dataset and all relevant information
    :param name: name of the dataset
    :param save: if set to True, save the dataset as a pickle file. Defaults to False
    :return: Preprocessed dataset and relevant information
    """
    def get_numpy(df):
        new_df = df.copy()
        cat_columns = new_df.select_dtypes(['category']).columns
        new_df[cat_columns] = new_df[cat_columns].apply(lambda x: x.cat.codes)
        return new_df.values

    if name == 'adult':
        # Load data
        feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', \
                         'marital-status', 'occupation', 'relationship', 'race', 'sex', \
                         'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
        df = pd.read_csv('../dataset/adult.data', names=feature_names)
        if verbose:
            print('Raw Dataset loaded.')
        num_train = df.shape[0]
        pos_class_label = ' >50K'
        neg_class_label = ' <=50K'
        y = np.zeros(num_train)
        y[df.iloc[:,-1].values == pos_class_label] = 1
#         df = df.drop(['fnlwgt', 'education-num'], axis=1)
        df = df.drop(['fnlwgt'], axis=1)
        num_var_names = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
#         cat_var_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        cat_var_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
        feature_names = num_var_names + cat_var_names
        df = df[feature_names]
        df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship'], prefix_sep='=')
        if verbose:
            print('Selecting relevant features complete.')

        for col in df:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(float)

        dtypes = df.dtypes

        X = get_numpy(df)
        if verbose:
            print('Numpy conversion complete.')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        if verbose:
            print('Dataset split complete.')

        # sens idx
        race_idx = df.columns.get_loc('race')
        sex_idx = df.columns.get_loc('sex')
        print( df.columns.get_loc('sex'))
        sens_idc = [race_idx, sex_idx]

        race_cats = df[feature_names[race_idx]].cat.categories
        sex_cats = df[feature_names[sex_idx]].cat.categories
        if verbose:
            print(race_cats, sex_cats)

        # Remove sensitive information from data
        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

        if verbose:
            print('Senstive attribute removal complete.')

    elif name == 'bank':
        # Load data
        
        df = pd.read_csv('../dataset/bank.csv', sep = ';', na_values=['unknown'])

        df['age'] = df['age'].apply(lambda x: x >= 25)
        df = df[np.array(df.default == 'no') + np.array(df.default == 'yes')]

        #         num_var_names = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
        cat_var_names = ['job', 'marital', 'education', 'default',
                         'housing', 'loan', 'contact', 'month', 'day_of_week',
                     'poutcome']
        #         feature_names = num_var_names + cat_var_names
        #         df = df[feature_names]

#         df = df.drop(['default'], axis=1)
        df = pd.get_dummies(df, columns=cat_var_names, prefix_sep='=')



        for col in df:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(float)

        

        Xy = get_numpy(df)


        idx = np.zeros(Xy.shape[-1]).astype(bool)
        idx[df.columns.get_loc('y')] = 1

        X = Xy[:, ~idx]
        y = Xy[:, idx].reshape(-1)       

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        
        dtypes = df.dtypes[~idx]

        # sens idx
        sex_idx = df.columns.get_loc('sex')
        race_idx = df.columns.get_loc('race')
        sens_idc = [sex_idx]


        # Remove sensitive information from data
        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

        if verbose:
            print('Senstive attribute removal complete.')
            
    elif name == 'compas':
        
        def quantizeLOS(x):
            if x<= 7:
                return '<week'
            if 8<x<=93:
                return '<3months'
            else:
                return '>3 months'


                # Load data
        df = pd.read_csv('../dataset/compas.csv', index_col='id', na_values=[])

        df = df[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'juv_fel_count',
                  'juv_misd_count', 'juv_other_count', 'sex', 'priors_count', 'days_b_screening_arrest', 'decile_score','c_charge_desc',
                 'two_year_recid', 'c_jail_in', 'c_jail_out']]

        # Indices of data samples to keep
        ix = df['days_b_screening_arrest'] <= 30
        ix = (df['days_b_screening_arrest'] >= -30) & ix
        # ix = (df['is_recid'] != -1) & ix
        ix = (df['c_charge_degree'] != "O") & ix
        ix = (df['score_text'] != 'N/A') & ix
        df = df.loc[ix,:]
        df['length_of_stay'] = abs(pd.to_datetime(df['c_jail_out'])-
                                pd.to_datetime(df['c_jail_in'])).apply(
                                                        lambda x: x.days)

        # Restrict races to African-American and Caucasian
        dfcut = df.loc[~df['race'].isin(['Native American','Hispanic','Asian','Other']),:]

        # Restrict the features to use
        dfcutQ = dfcut[['sex','race','age_cat','c_charge_degree','score_text','priors_count',
                'length_of_stay', 'two_year_recid','c_charge_desc']].copy()
        dfcutQ['length_of_stay'] = dfcutQ['length_of_stay'].apply(lambda x: quantizeLOS(x))

        num_train = dfcutQ.shape[0]

        num_var_names = ['two_year_recid', 'sex','race', 'score_text','priors_count', 'length_of_stay','c_charge_desc' ]
        categorical_features = ['age_cat','c_charge_degree']

        dfcutQ = pd.get_dummies(dfcutQ, columns=categorical_features, prefix_sep='=')

        race_idx = dfcutQ.drop(['two_year_recid'], axis = 1).columns.get_loc('race')
        sex_idx = dfcutQ.drop(['two_year_recid'], axis = 1).columns.get_loc('sex')
        sens_idc = [race_idx, sex_idx]

        for col in dfcutQ:
            if dfcutQ[col].dtype == 'object':
                dfcutQ[col] = dfcutQ[col].astype('category')
            else:
                dfcutQ[col] = dfcutQ[col].astype(float)


        pos_class_label = 1
        neg_class_label = 0

        idx = np.zeros(dfcutQ.shape[1]).astype(bool)
        y_idx = dfcutQ.columns.get_loc('two_year_recid')
        idx[y_idx] = True

        Xy = get_numpy(dfcutQ)

        X = Xy[:, ~idx]
        y = Xy[:, idx].reshape(-1)

        idx = X[:, 5] == -1

        X = X[~idx, :]
        y = y[~idx]

        dtypes = dfcutQ.drop(['two_year_recid'], axis = 1).dtypes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)

        if verbose:
            print('Dataset split complete.')

        race_cats = dfcutQ.iloc[:,dfcutQ.columns.get_loc('race')].cat.categories
        sex_cats = dfcutQ.iloc[:,dfcutQ.columns.get_loc('sex')].cat.categories
        if verbose:
            print(race_cats, sex_cats)

        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

        if verbose:
            print('Senstive attribute removal complete.')
    
    elif name == 'meps':
        
        df = pd.read_csv('../dataset/meps.csv', sep=',', na_values=[])
        
        def race(row):
            if ((row['HISPANX'] == 2) and (row['RACEV2X'] == 1)):  #non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
                return 'White'
            return 'Non-White'
        
        def sex(row):
            if row['SEX'] == 1:
                return 'female'
            return 'male'

        df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
        df['SEX'] = df.apply(lambda row: sex(row), axis=1)
        
        df = df.rename(columns = {'RACEV2X' : 'RACE'})

        df = df[df['PANEL'] == 19]

        df = df.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                              'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                              'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                              'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                              'POVCAT15' : 'POVCAT', 'INSCOV15' : 'INSCOV'})

        df = df[df['REGION'] >= 0] # remove values -1
        df = df[df['AGE'] >= 0] # remove values -1

        df = df[df['MARRY'] >= 0] # remove values -1, -7, -8, -9

        df = df[df['ASTHDX'] >= 0] # remove values -1, -7, -8, -9

        df = df[(df[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                                 'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]  #for all other categorical features, remove values < -1
                
        def utilization(row):
            return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

        df['TOTEXP15'] = df.apply(lambda row: utilization(row), axis=1)
        lessE = df['TOTEXP15'] < 10.0
        df.loc[lessE,'TOTEXP15'] = 0.0
        moreE = df['TOTEXP15'] >= 10.0
        df.loc[moreE,'TOTEXP15'] = 1.0

        df = df.rename(columns = {'TOTEXP15' : 'UTILIZATION'})
        
        features_to_keep=['REGION','AGE','SEX','RACE','MARRY',
                                 'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42','PCS42',
                                 'MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','UTILIZATION']
        
        categorical_features=['REGION','SEX', 'MARRY',
             'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
             'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
             'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
             'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
             'PHQ242','EMPST','POVCAT','INSCOV']
        
        df = df[features_to_keep]
        df = pd.get_dummies(df, columns=categorical_features, prefix_sep='=')
        
        num_train = df.shape[0]

        pos_class_label = 1
        neg_class_label = 0
        y = np.zeros(num_train)
        
        verbose = True
        
         # sens idx
        race_idx = df.columns.get_loc('RACE')
        sex_idx = 6
        sens_idc = [race_idx]
         
        for col in df:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(float)
                
        
        
        idx = np.zeros(df.shape[1]).astype(bool)
        y_idx = df.columns.get_loc('UTILIZATION')
        idx[y_idx] = True
        
#         min_max_scaler = MaxAbsScaler()
        Xy = get_numpy(df)
        X = Xy[:, ~idx]
        y = Xy[:, idx].reshape(-1)
        
        df = df.drop(['UTILIZATION'], axis=1)
        dtypes = df.dtypes
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        if verbose:
            print('Dataset split complete.')

        # Remove sensitive information from data
        
        race_cats = df['RACE'].cat.categories
#         sex_cats = df[feature_names[sex_idx]].cat.categories
        if verbose:
            print(race_cats)
            
        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]


        if verbose:
            print('Senstive attribute removal complete.')

    elif name == 'hmda':
        # Read raw data from csv
        df_raw = pd.read_csv('../dataset/hmda_2017_all.csv')
        print('Raw Dataset loaded.')

        # Extract useful features and separate them according to num/cat vars
        feature_names = ['loan_type_name','property_type_name', 'loan_purpose_name','owner_occupancy_name','loan_amount_000s', \
                         'preapproval_name', 'msamd_name', 'state_name', \
                         'county_name', 'applicant_race_name_1', 'applicant_sex_name', \
                         'applicant_income_000s', 'purchaser_type_name', 'lien_status_name', \
                         'population', 'minority_population', 'hud_median_family_income', 'tract_to_msamd_income', \
                         'number_of_owner_occupied_units', 'number_of_1_to_4_family_units']
        label_name = 'action_taken_name'
        num_train = df_raw.shape[0]
        y_raw = df_raw[label_name].values
        num_var_names = [x for x in feature_names if df_raw[x].dtypes == 'float64']
        cat_var_names = [x for x in feature_names if df_raw[x].dtypes != 'float64']
        feature_names = num_var_names + cat_var_names
        print(len(num_var_names), len(cat_var_names))
        df = df_raw[num_var_names + cat_var_names]
        print(num_var_names + cat_var_names)
        print('Selecting relevant features complete.')

        # Define what the labels are
        pos_labels = ['Loan originated', 'Loan purchased by the institution']
        neg_labels = ['Application approved but not accepted',
                      'Application denied by financial institution',
                      'Preapproval request denied by financial institution',
                      'Preapproval request approved but not accepted']

        pos_idx = np.array([])
        for l in pos_labels:
            pos_idx = np.concatenate((np.where(y_raw == l)[0], pos_idx))

        neg_idx = np.array([])
        for l in neg_labels:
            neg_idx = np.concatenate((np.where(y_raw == l)[0], neg_idx))

        X_pos = df.iloc[pos_idx][feature_names]
        X_neg = df.iloc[neg_idx][feature_names]
        # Remove rows with nan vals
        X_pos = X_pos.dropna(axis='rows')
        X_neg = X_neg.dropna(axis='rows')

        # Concat pos and neg samples
        Xdf = pd.concat((X_pos, X_neg))

        # Create labels
        y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
        for col in Xdf:
            if Xdf[col].dtype == 'object':
                Xdf[col] = Xdf[col].astype('category')
            else:
                Xdf[col] = Xdf[col].astype(np.float64)

        dtypes = Xdf.dtypes

        # get numpy format
        X = get_numpy(Xdf)

        print('Numpy conversion complete.')

        # split train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)

        print('Dataset split complete.')

        race_idx = 16
        sex_idx = 17

        race_cats = Xdf[feature_names[race_idx]].cat.categories
        sex_cats = Xdf[feature_names[sex_idx]].cat.categories

        #sensitive features
        sens_idc = [race_idx, sex_idx]

        # refer to these categorical tables for setting the index for positive and negative groups w.r.t senstive attr.
        print(race_cats, sex_cats)

        # Remove sensitive information from data
        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

        print('Sensitive attribute removal complete.')

    elif name == 'synth':
        def sample_from_gaussian(pos_mean,
                                 pos_cov,
                                 neg_mean,
                                 neg_cov,
                                 angle=np.pi/3,
                                 n_pos=200,
                                 n_neg=200,
                                 seed=0,
                                 corr_sens=True):
            np.random.seed(seed)
            x_pos = np.random.multivariate_normal(pos_mean, pos_cov, n_pos)
            np.random.seed(seed)
            x_neg = np.random.multivariate_normal(neg_mean, neg_cov, n_neg)
            X = np.vstack((x_pos, x_neg))
            y = np.hstack((np.ones(n_pos), np.zeros(n_neg)))
            n = y.shape[0]
            if corr_sens:
                # correlated sens data
                rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                xp = np.dot(X, rot)
                p1 = scp.stats.multivariate_normal.pdf(xp, pos_mean, pos_cov)
                p0 = scp.stats.multivariate_normal.pdf(xp, neg_mean, neg_cov)
                p = p1 / (p1 + p0)
                np.random.seed(seed)
                sens_attr = scp.stats.bernoulli.rvs(p)
            else:
                # independent sens data
                np.random.seed(seed)
                sens_attr = np.random.binomial(1, 0.5, n)
            return X, y, sens_attr

        ## NOTE change these variables for different distribution/generation of synth data.
        pos_mean = np.array([2,2])
        pos_cov = np.array([[5, 1], [1,5]])
        neg_mean = np.array([-2,-2])
        neg_cov = np.array([[10, 1],[1, 3]])
        n_pos = 500
        n_neg = 300
        angle = np.pi / 2
        #corr_sens = False
        X, y, sens = sample_from_gaussian(pos_mean,
                                          pos_cov,
                                          neg_mean,
                                          neg_cov,
                                          angle=angle,
                                          n_pos=n_pos,
                                          n_neg=n_neg,
                                          corr_sens=corr_sens)
        X = np.concatenate((X, np.expand_dims(sens, 1)), axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        dtypes = None
        dtypes_ = None
        sens_idc = [2]
        X_train_removed = X_train[:,:2]
        X_test_removed = X_test[:,:2]
        race_idx = None
        sex_idx = None

    elif name == 'synth2':
        def sample_from_gaussian(pos_mean,
                                 pos_cov,
                                 neg_mean,
                                 neg_cov,
                                 thr=0,
                                 n_pos=200,
                                 n_neg=200,
                                 seed=0,
                                 corr_sens=True):
            np.random.seed(seed)
            x_pos = np.random.multivariate_normal(pos_mean, pos_cov, n_pos)
            np.random.seed(seed)
            x_neg = np.random.multivariate_normal(neg_mean, neg_cov, n_neg)
            X = np.vstack((x_pos, x_neg))
            y = np.hstack((np.ones(n_pos), np.zeros(n_neg)))
            n = y.shape[0]
            if corr_sens:
                # correlated sens data
                sens_attr = np.zeros(n)
                idx = np.where(X[:,0] > thr)[0]
                sens_attr[idx] = 1
            else:
                # independent sens data
                np.random.seed(seed)
                sens_attr = np.random.binomial(1, 0.5, n)
            return X, y, sens_attr

        ## NOTE change these variables for different distribution/generation of synth data.
        pos_mean = np.array([2,2])
        pos_cov = np.array([[5, 1], [1,5]])
        neg_mean = np.array([-2,-2])
        neg_cov = np.array([[10, 1],[1, 3]])
        n_pos = 500
        n_neg = 300
        thr = 0
        #corr_sens = False
        X, y, sens = sample_from_gaussian(pos_mean,
                                          pos_cov,
                                          neg_mean,
                                          neg_cov,
                                          thr=thr,
                                          n_pos=n_pos,
                                          n_neg=n_neg,
                                          corr_sens=corr_sens)
        X = np.concatenate((X, np.expand_dims(sens, 1)), axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)
        dtypes = None
        dtypes_ = None
        sens_idc = [2]
        X_train_removed = X_train[:,:2]
        X_test_removed = X_test[:,:2]
        race_idx = None
        sex_idx = None

    elif name == 'german':
        # Download data if needed
        _german_loan_attribute_map = dict(
            A11='< 0 DM',
            A12='0-200 DM',
            A13='>= 200 DM',
            A14='no checking',
            A30='no credits',
            A31='all credits paid back',
            A32='existing credits paid back',
            A33='delayed past payments',
            A34='critical account',
            A40='car (new)',
            A41='car (used)',
            A42='furniture/equipment',
            A43='radio/television',
            A44='domestic appliances',
            A45='repairs',
            A46='education',
            A47='(vacation?)',
            A48='retraining',
            A49='business',
            A410='others',
            A61='< 100 DM',
            A62='100-500 DM',
            A63='500-1000 DM',
            A64='>= 1000 DM',
            A65='unknown/no sav acct',
            A71='unemployed',
            A72='< 1 year',
            A73='1-4 years',
            A74='4-7 years',
            A75='>= 7 years',
            #A91='male & divorced',
            #A92='female & divorced/married',
            #A93='male & single',
            #A94='male & married',
            #A95='female & single',
            A91='male',
            A92='female',
            A93='male',
            A94='male',
            A95='female',
            A101='none',
            A102='co-applicant',
            A103='guarantor',
            A121='real estate',
            A122='life insurance',
            A123='car or other',
            A124='unknown/no property',
            A141='bank',
            A142='stores',
            A143='none',
            A151='rent',
            A152='own',
            A153='for free',
            A171='unskilled & non-resident',
            A172='unskilled & resident',
            A173='skilled employee',
            A174='management/self-employed',
            A191='no telephone',
            A192='has telephone',
            A201='foreigner',
            A202='non-foreigner',
        )

        filename = 'german.data'
        if not os.path.isfile(filename):
            print('Downloading data to %s' % os.path.abspath(filename))
            urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                                       filename)

        # Load data and setup dtypes
        col_names = [
            'checkin_acc', 'duration', 'credit_history', 'purpose', 'amount',
            'saving_acc', 'present_emp_since', 'inst_rate', 'personal_status',
            'other_debtors', 'residing_since', 'property', 'age',
            'inst_plans', 'housing', 'num_credits',
            'job', 'dependents', 'telephone', 'foreign_worker', 'status']
        
#         AIF360
        column_names = ['status', 'month', 'credit_history',
            'purpose', 'credit_amount', 'savings', 'employment',
            'investment_as_income_percentage', 'personal_status',
            'other_debtors', 'residence_since', 'property', 'age',
            'installment_plans', 'housing', 'number_of_credits',
            'skill_level', 'people_liable_for', 'telephone',
            'foreign_worker', 'credit']
        
        df = pd.read_csv(filename, delimiter=' ', header=None, names=column_names)
        
        def group_credit_hist(x):
            if x in ['A30', 'A31', 'A32']:
                return 'None/Paid'
            elif x == 'A33':
                return 'Delay'
            elif x == 'A34':
                return 'Other'
            else:
                return 'NA'

        def group_employ(x):
            if x == 'A71':
                return 'Unemployed'
            elif x in ['A72', 'A73']:
                return '1-4 years'
            elif x in ['A74', 'A75']:
                return '4+ years'
            else:
                return 'NA'

        def group_savings(x):
            if x in ['A61', 'A62']:
                return '<500'
            elif x in ['A63', 'A64']:
                return '500+'
            elif x == 'A65':
                return 'Unknown/None'
            else:
                return 'NA'
        def group_status(x):
            if x in ['A11', 'A12']:
                return '<200'
            elif x in ['A13']:
                return '200+'
            elif x == 'A14':
                return 'None'
            else:
                return 'NA'

        def group_purpose(x):
            if x in ['A40', 'A41', 'A42', 'A43', 'A47', 'A410']:
                return 'non-essential'
            elif x in ['A44', 'A45', 'A46', 'A48', 'A49']:
                return 'essential'

        status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                      'A92': 'female', 'A95': 'female'}
        df['sex'] = df['personal_status'].replace(status_map)

        # group credit history, savings, and employment
        df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
        df['savings'] = df['savings'].apply(lambda x: group_savings(x))
        df['employment'] = df['employment'].apply(lambda x: group_employ(x))
    #     df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
        df['purpose'] = df['purpose'].apply(lambda x:group_purpose(x))
        df['status'] = df['status'].apply(lambda x: group_status(x))
        
        cat_features = ['credit_history', 'savings', 'employment',  'purpose', 'other_debtors', 'property', 'housing', 'skill_level', \
                'investment_as_income_percentage', 'status', 'installment_plans', 'foreign_worker']
        
        df = pd.get_dummies(df, columns=cat_features, prefix_sep='=')
        df = df.drop(['telephone', 'personal_status',], axis = 1)
    
            
        for col in df:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(float)
                
        df['age'] = df['age'].apply(lambda x: x >= 25).astype('category')

        def get_numpy(df):
            new_df = df.copy()
            cat_columns = new_df.select_dtypes(['category']).columns
            new_df[cat_columns] = new_df[cat_columns].apply(lambda x: x.cat.codes)
            return new_df.values
        
        y_idx = df.columns.get_loc('credit')
        idx = np.zeros(df.shape[1]).astype(bool)
        idx[y_idx] = True
        
        Xy = get_numpy(df)
        X = Xy[:,~idx]
        y = Xy[:,idx].reshape(-1)
        
        # Make 1 (good customer) and 0 (bad customer)
        # (Originally 2 is bad customer and 1 is good customer)
        sel_bad = y == 2
        y[sel_bad] = 0
        y[~sel_bad] = 1
        feature_labels = df.columns.values[:-1]  # Last is prediction
        dtypes = df.dtypes[~idx]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=seed)

        # Senstivie attribute
#         foreign = 19
        age_idx = df.columns.get_loc('age')
        sex_idx = df.columns.get_loc('sex')
        sens_idc = [sex_idx, age_idx]

        
        
        age_cats = df['age'].cat.categories
        sex_cats = df['sex'].cat.categories
        print(age_cats, sex_cats)

        # Remove sensitive information from data
        X_train_removed = np.delete(X_train, sens_idc , 1)
        X_test_removed = np.delete(X_test, sens_idc , 1)
        dtypes_ = dtypes[np.delete(np.arange(X.shape[1]), sens_idc, 0)]

        race_idx = age_idx

    else:
        raise ValueError('Data name invalid.')

    return X_train, y_train, X_test, y_test, X_train_removed, X_test_removed, dtypes, dtypes_, sens_idc, race_idx, sex_idx

def get_csv_eqodd(fm, data_name='adult'):
    assert(fm.model is not None)
    # get csv files required for the eq_odds code to run
    label = fm.y_test
    group = fm.X_test[:, fm.sens_idx]
    prediction = fm.model.predict_proba(fm.X_test_removed)[:,1] # positive label prediction
    # make csv file
    f = open('%s_predictions.csv'%data_name, 'w')
    f.write(',label,group,prediction\n')
    for i, e in enumerate(zip(label, group, prediction)):
        line = '%d,%0.2f,%0.2f,%f\n'%(i, e[0],  e[1], e[2])
        f.write(line)
    f.close()


    
