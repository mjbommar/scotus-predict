''''
@author Michael J Bommarito II; michael@bommaritollc.com
@date 20140430

Machine learning model/training setup
'''

# Imports
import copy
import numpy
import pandas
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.ensemble
import sklearn.grid_search


# SCOTUS imports
from scotus.utility import get_data_by_condition, get_data_before_date, get_means

# Roll windows
ROLL_WINDOWS=[None, 10]

# Setup hyperparameters
search_parameters = {'feature_select__percentile': [100],
                     'classify__min_samples_leaf': [2],
                         'classify__max_depth': [32],
                         'classify__max_features': [24],
                         'classify__n_estimators': [4000]
                         }

# Set the list of verbatim features to copy from the case and docket variables
raw_court_features = {'lc_direction': 'lcDispositionDirection',
                    'admin_action': 'adminAction', 
                    'case_origin': 'caseOrigin',
                    'case_source': 'caseSource',
                    'case_origin_circuit': 'caseOrigin_circuit',
                    'case_source_circuit': 'caseSource_circuit',
                    'cert_reason': 'certReason',
                    'issue': 'issue',
                    'issue_area': 'issueArea',
                    'month_decision': 'monthDecision',
                    'month_argument': 'monthArgument',
                    'jurisdiction': 'jurisdiction',
                    'law_type': 'lawType',
                    'lc_disagreement': 'lcDisagreement', 
                    'lc_disposition': 'lcDisposition',
                    'natural_court': 'naturalCourt',
                    'petitioner': 'petitioner',
                    'respondent': 'respondent',
                    'petitioner_dk': 'petitioner_dk',
                    'respondent_dk': 'respondent_dk',
                    }

raw_justice_features = {'justice': 'justice',
                        'direction': 'direction',
                        'gender':'gender',
                        'year_of_birth':'year_of_birth',
                        'party_president': 'party_president',
                        'segal_cover':'segal_cover',
                        'is_chief': 'is_chief'}



def train_model(feature_data, target_data, search_parameters):
    '''
    Train a model.
    '''
    # Train model|
    model_pipeline = sklearn.pipeline.Pipeline([
            ('scale', sklearn.preprocessing.StandardScaler()),
            ('feature_select', sklearn.feature_selection.SelectPercentile(sklearn.feature_selection.f_classif)),
            ('classify', sklearn.ensemble.ExtraTreesClassifier(bootstrap=True, 
                                                               criterion='entropy',
                                                               ))
        ])
    
    # Create the stratified cross-validation folder
    cv = sklearn.cross_validation.StratifiedKFold(target_data, n_folds=10)
    
    # Create grid searcher
    grid_search = sklearn.grid_search.GridSearchCV(model_pipeline,
                                                   search_parameters,
                                                   scoring=sklearn.metrics.make_scorer(sklearn.metrics.fbeta_score, beta=1.0,
                                                                                       pos_label=pandas.Series(target_data)\
                                                                                           .value_counts().idxmax()),
                                                   cv=cv,
                                                   verbose=0,
                                                   n_jobs=2)

    # Fit model in grid search
    grid_search.fit(feature_data, target_data)

    return grid_search


def get_ml_row(docket_id, scdb_case_data, scdb_justice_data):
    '''
    Get a single row of feature and target data for a given docket ID.
    '''
    '''
    A block features: code block case-derived features
    '''
    # Get the case and justice level data
    docket_case_data = scdb_case_data.ix[scdb_case_data['docketId'] == docket_id]
    docket_justice_data = scdb_justice_data.ix[scdb_case_data['docketId'] == docket_id]
    docket_justice_list = list(set(docket_justice_data['justice'].tolist()))

    # Get the data prior to this docket.
    docket_decision_date = docket_case_data['dateDecision'].tolist().pop()
    docket_before_case_data = get_data_before_date(docket_decision_date, scdb_case_data)
    docket_before_justice_data = get_data_before_date(docket_decision_date, scdb_justice_data)
    
    # Get some relevant information for conditioning
    docket_circuit_origin = docket_case_data['caseOrigin_circuit'].tolist().pop()
    docket_circuit_source = docket_case_data['caseSource_circuit'].tolist().pop()
    docket_issue = docket_case_data['issue'].tolist().pop()
    docket_petitioner = docket_case_data['petitioner'].tolist().pop()
    docket_respondent = docket_case_data['respondent'].tolist().pop()
    
    # Info
    '''
    print((docket_id, docket_decision_date, docket_before_case_data.shape[0],
           docket_before_justice_data.shape[0]))
    '''
    
    # Get unconditioned direction data by court
    court_direction_data = get_means(docket_before_case_data, 'decisionDirection', 'court_direction',
          windows=ROLL_WINDOWS)
    
    # Get conditioned direction data for Supreme Court
    court_issue_data = get_data_by_condition(docket_issue, 'issue', docket_before_case_data)
    court_direction_issue_data = get_means(court_issue_data, 'decisionDirection', 'court_direction_issue',
                                                      windows=ROLL_WINDOWS)
    
    court_circuit_origin_data = get_data_by_condition(docket_circuit_origin, 'caseOrigin_circuit',
                                                      docket_before_case_data)
    court_direction_circuit_origin_data = get_means(court_circuit_origin_data, 'decisionDirection', 
                                                    'court_direction_circuit_origin', windows=ROLL_WINDOWS)
    
    court_circuit_source_data = get_data_by_condition(docket_circuit_source, 'caseSource_circuit',
                                                      docket_before_case_data)
    court_direction_circuit_source_data = get_means(court_circuit_source_data, 'decisionDirection', 
                                                    'court_direction_circuit_source', windows=ROLL_WINDOWS)
    
    court_petitioner_data = get_data_by_condition(docket_petitioner, 'petitioner',
                                                      docket_before_case_data)
    court_direction_petitioner_data = get_means(court_petitioner_data, 'decisionDirection', 
                                                    'court_direction_petitioner', windows=ROLL_WINDOWS)
    
    court_respondent_data = get_data_by_condition(docket_respondent, 'respondent',
                                                      docket_before_case_data)
    court_direction_respondent_data = get_means(court_respondent_data, 'decisionDirection', 
                                                    'court_direction_respondent', windows=ROLL_WINDOWS)
    
    # Get conditioned direction data for lower court
    '''
    We are calculating the mean direction of the lower court's disposition conditioned on:
      1. Issue
      2. Court of source
      3. Petitioner/respondent
      
    Note that these are for ALL lower courts, other than #2, which constrain the lower court to the specific
    court of origin or source.
    '''
    lower_court_issue_data = get_data_by_condition(docket_issue, 'issue', docket_before_case_data)
    lower_court_direction_issue_data = get_means(court_issue_data, 'lcDispositionDirection', 
                                                 'lower_court_direction_issue', windows=ROLL_WINDOWS)
    
    lower_court_circuit_source_data = get_data_by_condition(docket_circuit_source, 'caseSource_circuit',
                                                      docket_before_case_data)
    lower_court_direction_circuit_source_data = get_means(court_circuit_source_data, 'lcDispositionDirection', 
                                                    'lower_court_direction_circuit_source', windows=ROLL_WINDOWS)

    lower_court_petitioner_data = get_data_by_condition(docket_petitioner, 'petitioner',
                                                      docket_before_case_data)
    lower_court_direction_petitioner_data = get_means(court_petitioner_data, 'lcDispositionDirection', 
                                                    'lower_court_direction_petitioner', windows=ROLL_WINDOWS)
    
    lower_court_respondent_data = get_data_by_condition(docket_respondent, 'respondent',
                                                      docket_before_case_data)
    lower_court_direction_respondent_data = get_means(court_respondent_data, 'lcDispositionDirection', 
                                                    'lower_court_direction_respondent', windows=ROLL_WINDOWS)

    # Justice data rows
    justice_dict_rows = []
    justice_dict = {}
    
    # Aggregate all features
    all_court_features = dict()

    # Add the raw features and indices
    all_court_features['docket'] = docket_id
    for feature_name in raw_court_features:
        all_court_features[feature_name] = docket_case_data[raw_court_features[feature_name]].tolist().pop()

    # Add the court level features
    all_court_features.update(court_direction_data)
    all_court_features.update(court_direction_issue_data)
    all_court_features.update(court_direction_circuit_origin_data)  
    all_court_features.update(court_direction_circuit_source_data)
    all_court_features.update(court_direction_petitioner_data)
    all_court_features.update(court_direction_respondent_data)
    all_court_features.update(lower_court_direction_issue_data)
    all_court_features.update(lower_court_direction_circuit_source_data)
    all_court_features.update(lower_court_direction_petitioner_data)
    all_court_features.update(lower_court_direction_respondent_data)
    
    # Iterate over all justices
    for justice_group_name, justice_group_data in docket_before_justice_data.groupby('justice'):
        '''
        B block features: code block justice-derived features
        '''
        # Skip the justice ID
        if justice_group_name not in docket_justice_list:
            continue

        # Get unconditioned justice direction and justice-court agreement data
        justice_direction_data = get_means(justice_group_data, 'direction', 
              'justice_direction', windows=ROLL_WINDOWS)
        justice_agree_data = get_means(justice_group_data, 'agree',
              'justice_agree', windows=ROLL_WINDOWS)
        
        # Get conditioned justice direction and justice-court agreement data
        justice_group_issue_data = get_data_by_condition(docket_issue, 'issue', 
                                                                  justice_group_data)
        justice_direction_issue_data = get_means(justice_group_issue_data,
                                                          'direction', 'justice_direction_issue',
                                                          windows=ROLL_WINDOWS)
        
        justice_group_circuit_origin_data = get_data_by_condition(docket_circuit_origin, 'caseOrigin_circuit', 
                                                                  justice_group_data)
        justice_direction_circuit_origin_data = get_means(justice_group_circuit_origin_data,
                                                          'direction', 'justice_direction_circuit_origin',
                                                          windows=ROLL_WINDOWS)
        
        justice_group_circuit_source_data = get_data_by_condition(docket_circuit_source, 'caseSource_circuit', 
                                                                  justice_group_data)
        justice_direction_circuit_source_data = get_means(justice_group_circuit_source_data,
                                                          'direction', 'justice_direction_circuit_source',
                                                          windows=ROLL_WINDOWS)
        
        justice_group_petitioner_data = get_data_by_condition(docket_petitioner, 'petitioner', 
                                                                  justice_group_data)
        justice_direction_petitioner_data = get_means(justice_group_petitioner_data,
                                                          'direction', 'justice_direction_petitioner',
                                                          windows=ROLL_WINDOWS)
        
        justice_group_respondent_data = get_data_by_condition(docket_respondent, 'respondent', 
                                                                  justice_group_data)
        justice_direction_respondent_data = get_means(justice_group_respondent_data,
                                                          'direction', 'justice_direction_respondent',
                                                          windows=ROLL_WINDOWS)

        # Roll up the justice data
        justice_dict = dict()
        justice_dict.update(all_court_features)        
        justice_dict.update(justice_direction_data)
        justice_dict.update(justice_agree_data)
        justice_dict.update(justice_direction_issue_data)
        justice_dict.update(justice_direction_circuit_origin_data)
        justice_dict.update(justice_direction_circuit_source_data)
        justice_dict.update(justice_direction_petitioner_data)
        justice_dict.update(justice_direction_respondent_data)
        justice_dict['justice'] = copy.copy(justice_group_name)
        for feature_name in raw_justice_features:
            justice_dict[feature_name] = docket_justice_data.ix[docket_justice_data['justice'] == justice_group_name]                [raw_justice_features[feature_name]].copy().tolist().pop()

        justice_dict_rows.append(copy.deepcopy(justice_dict))

    '''
    C block features: code block for case-and-justice-derived (second-level) features
    '''
    # Convert to data frame
    all_justice_features = pandas.DataFrame.from_dict(justice_dict_rows)
    
    # Current court means
    all_justice_features['current_court_direction_mean'] = all_justice_features['justice_direction_mean'].mean()
    all_justice_features['current_court_direction_std'] = all_justice_features['justice_direction_mean'].std()

    
    all_justice_features['current_court_agree_mean'] = all_justice_features['justice_agree_mean'].mean()
    all_justice_features['current_court_agree_std'] = all_justice_features['justice_agree_mean'].std()
    
    all_justice_features['current_court_direction_issue_mean'] = all_justice_features['justice_direction_issue_mean'].mean()
    all_justice_features['current_court_direction_issue_std'] = all_justice_features['justice_direction_issue_mean'].std()
    
    all_justice_features['current_court_direction_circuit_origin_mean'] = all_justice_features['justice_direction_circuit_origin_mean'].mean()        
    all_justice_features['current_court_direction_circuit_origin_std'] = all_justice_features['justice_direction_circuit_origin_mean'].std()
    
    all_justice_features['current_court_direction_circuit_source_mean'] = all_justice_features['justice_direction_circuit_source_mean'].mean()
    all_justice_features['current_court_direction_circuit_source_std'] = all_justice_features['justice_direction_circuit_source_mean'].std()
        
    all_justice_features['current_court_direction_petitioner_mean'] = all_justice_features['justice_direction_petitioner_mean'].mean()
    all_justice_features['current_court_direction_petitioner_std'] = all_justice_features['justice_direction_petitioner_mean'].std()
    
    all_justice_features['current_court_direction_respondent_mean'] = all_justice_features['justice_direction_respondent_mean'].mean()
    all_justice_features['current_court_direction_respondent_std'] = all_justice_features['justice_direction_respondent_mean'].std()
    
    '''
    D block features: code block for differences between derived features
    '''
    all_justice_features['diff_justice_court_direction'] = all_justice_features['justice_direction_mean'] - all_justice_features['current_court_direction_mean']
    all_justice_features['diff_justice_court_direction_abs'] = all_justice_features['diff_justice_court_direction'].abs()
    all_justice_features['diff_justice_court_direction_issue'] = all_justice_features['justice_direction_issue_mean'] - all_justice_features['current_court_direction_issue_mean']
    all_justice_features['diff_justice_court_direction_issue_abs'] = all_justice_features['diff_justice_court_direction_issue'].abs()
    all_justice_features['diff_justice_court_direction_petitioner'] = all_justice_features['justice_direction_petitioner_mean'] - all_justice_features['current_court_direction_petitioner_mean']
    all_justice_features['diff_justice_court_direction_petitioner_abs'] = all_justice_features['diff_justice_court_direction_petitioner'].abs()
    all_justice_features['diff_justice_court_direction_respondent'] = all_justice_features['justice_direction_respondent_mean'] - all_justice_features['current_court_direction_respondent_mean']
    all_justice_features['diff_justice_court_direction_respondent_abs'] = all_justice_features['diff_justice_court_direction_respondent'].abs()
    all_justice_features['diff_justice_lc_direction'] = all_justice_features['justice_direction_mean'] - all_justice_features['lc_direction']
    all_justice_features['diff_justice_lc_direction_abs'] = all_justice_features['diff_justice_lc_direction'].abs()
    all_justice_features['diff_court_lc_direction'] = all_justice_features['current_court_direction_mean'] - all_justice_features['lc_direction']
    all_justice_features['diff_court_lc_direction_abs'] = all_justice_features['diff_court_lc_direction'].abs()
    
    '''
    E block features: z-score scaling
    '''
    
    if all_justice_features['diff_justice_court_direction'].std() > 0:
        all_justice_features['diff_justice_court_direction_z'] = (all_justice_features['diff_justice_court_direction'] - all_justice_features['diff_justice_court_direction'].mean()) / (all_justice_features['diff_justice_court_direction'].std())
    else:
        all_justice_features['diff_justice_court_direction_z'] = -98
    
    if all_justice_features['diff_justice_court_direction_issue'].std() > 0:
        all_justice_features['diff_justice_court_direction_issue_z'] = (all_justice_features['diff_justice_court_direction_issue'] - all_justice_features['diff_justice_court_direction_issue'].mean()) / (all_justice_features['diff_justice_court_direction_issue'].std())
    else:
        all_justice_features['diff_justice_court_direction_issue_z'] = -98

    if all_justice_features['diff_justice_lc_direction'].std() > 0:
        all_justice_features['diff_justice_lc_direction_z'] = (all_justice_features['diff_justice_lc_direction'] - all_justice_features['diff_justice_lc_direction'].mean()) / (all_justice_features['diff_justice_lc_direction'].std())
    else:
        all_justice_features['diff_justice_lc_direction_z'] = -98
        
    if all_justice_features['diff_court_lc_direction'].std() > 0:
        all_justice_features['diff_court_lc_direction_z'] = (all_justice_features['diff_court_lc_direction'] - all_justice_features['diff_court_lc_direction'].mean()) / (all_justice_features['diff_court_lc_direction'].std())
    else:
        all_justice_features['diff_court_lc_direction_z'] = -98
        
    # Get the CASE outcome.
    #all_justice_features['docket_outcome'] = (all_justice_features['direction'] != all_justice_features['lc_direction']).apply(int)
    all_justice_features['outcome'] = (all_justice_features['direction'] != all_justice_features['lc_direction']).apply(int)
    all_justice_features = all_justice_features.replace(numpy.inf, -98).replace(-numpy.inf, -98)
    all_justice_features = all_justice_features.fillna(-99)
    
    # Finally, calculate and label the target.
    all_justice_targets = all_justice_features.copy()
    all_justice_targets = pandas.DataFrame(all_justice_targets['outcome'].tolist())
    all_justice_targets = all_justice_targets.fillna(0)
    
    return all_justice_features, all_justice_targets
