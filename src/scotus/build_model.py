''''
@author Michael J Bommarito II; michael@bommaritollc.com
@date 20140430

Predict justice votes at the Surpreme Court from 1946 to 2013!
'''

# Imports
import copy
import numpy
import os
import pandas
import pylab
import sklearn.metrics
import time

# SCOTUS imports
from scotus.constants import court_circuit_map, justice_name_map, party_map_data
from scotus.utility import get_date, get_gender, get_justice_name, get_month, \
    get_party_president, get_segal_cover, get_year_from_docket, get_year_of_birth, \
    map_circuit, map_party, get_data_before_date, get_data_before_term, get_data_by_condition, \
    get_data_through_date, get_data_through_term
from scotus.model import get_ml_row, train_model, search_parameters, raw_court_features, raw_justice_features

# Load SCDB CSV data
scdb_case_data = pandas.DataFrame.from_csv('data/SCDB_2013_01_caseCentered_Citation.csv')
scdb_justice_data = pandas.DataFrame.from_csv('data/SCDB_2013_01_justiceCentered_Citation.csv')

'''
Apply date transforms to the data.
'''
# Preprocess some fields
scdb_case_data['dateDecision'] = scdb_case_data['dateDecision'].apply(get_date)
scdb_justice_data['dateDecision'] = scdb_justice_data['dateDecision'].apply(get_date)
scdb_case_data['dateArgument'] = scdb_case_data['dateArgument'].apply(get_date)
scdb_justice_data['dateArgument'] = scdb_justice_data['dateArgument'].apply(get_date)
scdb_case_data['dateRearg'] = scdb_case_data['dateRearg'].apply(get_date)
scdb_justice_data['dateRearg'] = scdb_justice_data['dateRearg'].apply(get_date)
scdb_case_data['monthDecision'] = scdb_case_data['dateDecision'].apply(get_month)
scdb_justice_data['monthDecision'] = scdb_justice_data['dateDecision'].apply(get_month)
scdb_case_data['monthArgument'] = scdb_case_data['dateArgument'].apply(get_month)
scdb_justice_data['monthArgument'] = scdb_justice_data['dateArgument'].apply(get_month)

'''
Apply other basic transforms to the data.
'''
# Set unspecified decision directions to None
scdb_case_data.loc[scdb_case_data['decisionDirection'] == 3, 'decisionDirection'] = 1.5
scdb_justice_data.loc[scdb_justice_data['decisionDirection'] == 3, 'decisionDirection'] = 1.5

# Map case origin and source
scdb_case_data['caseOrigin_circuit'] = scdb_case_data['caseOrigin'].apply(map_circuit)
scdb_justice_data['caseOrigin_circuit'] = scdb_justice_data['caseOrigin'].apply(map_circuit)
scdb_case_data['caseSource_circuit'] = scdb_case_data['caseSource'].apply(map_circuit)
scdb_justice_data['caseSource_circuit'] = scdb_justice_data['caseSource'].apply(map_circuit)

# Map party type
scdb_case_data['petitioner_dk'] = scdb_case_data['petitioner'].apply(map_party)
scdb_case_data['respondent_dk'] = scdb_case_data['respondent'].apply(map_party)
scdb_justice_data['petitioner_dk'] = scdb_justice_data['petitioner'].apply(map_party)
scdb_justice_data['respondent_dk'] = scdb_justice_data['respondent'].apply(map_party)

# Overturn
scdb_case_data['decisionOverturn'] = numpy.abs(numpy.sign(scdb_case_data['decisionDirection']                                                           - scdb_case_data['lcDispositionDirection']))
scdb_justice_data['decisionOverturn'] = numpy.abs(numpy.sign(scdb_justice_data['direction']                                                              - scdb_justice_data['lcDispositionDirection']))

# Handle the agreement field
scdb_justice_data['agree'] = (scdb_justice_data['direction'] == scdb_justice_data['decisionDirection'])

# Map data
scdb_justice_data['gender'] = scdb_justice_data['justice'].apply(get_gender)
scdb_justice_data['year_of_birth'] = scdb_justice_data['justice'].apply(get_year_of_birth)
scdb_justice_data['party_president'] = scdb_justice_data['justice'].apply(get_party_president)
scdb_justice_data['segal_cover'] = scdb_justice_data['justice'].apply(get_segal_cover)
scdb_justice_data['is_chief'] = [int(x.endswith(y)) for x, y, in zip(scdb_justice_data['justiceName'].tolist(),
                                                                     scdb_justice_data['chief'].tolist())]

# Sort cases by decision date and set into case list
docket_list = scdb_case_data.sort('dateDecision')['docketId'].tolist()

'''
Clean up unspecifiable/direction=3 values.
'''
scdb_case_data.loc[scdb_case_data['lcDispositionDirection'] == 3, 'lcDispositionDirection'] = 1.5
scdb_justice_data.loc[scdb_justice_data['lcDispositionDirection'] == 3, 'lcDispositionDirection'] = 1.5
scdb_case_data.loc[scdb_case_data['decisionDirection'] == 3, 'decisionDirection'] = 1.5
scdb_justice_data.loc[scdb_justice_data['decisionDirection'] == 3, 'decisionDirection'] = 1.5

# Set minimum record count
min_record_count = 100
max_record_count = 99999

# Setup total feature and target data
feature_data = pandas.DataFrame()
target_data = pandas.DataFrame()

# Setup the model
model = None
bad_feature_labels = ['docket',
                      'outcome',
                      'docket_outcome',
                      'case_outcome',
                      'disposition_outcome',
                      'direction',
                      #'admin_action',
                      #'case_origin', 'case_source', 
                      #'issue',
                      #'petitioner', 'respondent'
                      #'petitioner_dk', 'respondent_dk'
                      ]
feature_labels = []

# Outcome data
outcome_data = pandas.DataFrame()
case_outcome_data = pandas.DataFrame()

# Track the less likely label
min_label = 1.0

# Iterate over all dockets
num_dockets = 0
for docket_id in docket_list:
    # Increment dockets seen
    num_dockets += 1
    
    if max_record_count != None and num_dockets > max_record_count:
        break
    
    # Get rows of feature and target data
    feature_rows, target_rows = get_ml_row(docket_id, scdb_case_data, scdb_justice_data)
    
    # Now append to the feature and target lists
    feature_data = feature_data.append(feature_rows.copy())
    target_data = target_data.append(target_rows.copy())
    
    # Now re-calculate all the z-scaled values
    feature_data['justice_direction_mean_z'] = (feature_data['justice_direction_mean'] - pandas.expanding_mean(feature_data['justice_direction_mean']))                     / pandas.expanding_std(feature_data['justice_direction_mean'])
    feature_data['diff_justice_lc_direction_abs_z'] = (feature_data['diff_justice_lc_direction_abs'] - pandas.expanding_mean(feature_data['diff_justice_lc_direction_abs']))                     / pandas.expanding_std(feature_data['diff_justice_lc_direction_abs'])
    feature_data['diff_justice_lc_direction_z'] = (feature_data['diff_justice_lc_direction'] - pandas.expanding_mean(feature_data['diff_justice_lc_direction']))                     / pandas.expanding_std(feature_data['diff_justice_lc_direction'])
    feature_data['diff_court_lc_direction_abs_z'] = (feature_data['diff_court_lc_direction_abs'] - pandas.expanding_mean(feature_data['diff_court_lc_direction_abs']))                     / pandas.expanding_std(feature_data['diff_court_lc_direction_abs'])
    feature_data['justice_direction_issue_mean_z'] = (feature_data['justice_direction_issue_mean'] - pandas.expanding_mean(feature_data['justice_direction_issue_mean']))                     / pandas.expanding_std(feature_data['justice_direction_issue_mean'])
    feature_data['current_court_direction_issue_mean_z'] = (feature_data['current_court_direction_issue_mean'] - pandas.expanding_mean(feature_data['current_court_direction_issue_mean']))                     / pandas.expanding_std(feature_data['current_court_direction_issue_mean'])
    feature_data = feature_data.replace(-numpy.inf, -98)
    feature_data = feature_data.replace(numpy.inf, -98)
    feature_data = feature_data.fillna(-99)

    # Update any missing columns in E-block
    feature_rows = feature_data.ix[feature_data['docket'] == docket_id].sort('justice').copy()
    target_rows = feature_rows['outcome']
    
    # Check to see if we've trained a model yet.
    if model != None:
        # If so, let's test it.
        docket_outcome_data = feature_rows.copy()
        docket_outcome_data['prediction'] = model.predict(feature_rows[feature_labels])
        docket_outcome_data['target'] = target_rows.copy()
        
        # Get the vote of the court aggregated
        vote_mean_outcome = docket_outcome_data['prediction'].value_counts().idxmax()
        
        docket_outcome_data['docket_vote_mean'] = vote_mean_outcome
        docket_outcome_data['docket_vote_sum'] = docket_outcome_data['prediction'].sum() 
        
        # Append data to the case outcome data frame
        case_record = scdb_case_data.ix[scdb_case_data['docketId'] == docket_id]
        case_outcome_record = docket_outcome_data.ix[0][['docket', 'docket_outcome', 'docket_vote_mean', 'docket_vote_sum']]
        case_outcome_record['docket_outcome'] = int((case_record['lcDispositionDirection'] == case_record['decisionDirection']).tolist().pop())
        case_outcome_data = case_outcome_data.append(case_outcome_record)
        
        # Aggregate all data
        outcome_data = outcome_data.append(copy.deepcopy(docket_outcome_data))
        
        if num_dockets % 100 == 0:
            # Output the rolling confusion matrix every few ticks
            print(sklearn.metrics.classification_report(outcome_data['target'].tolist(),
                                              outcome_data['prediction'].tolist()))
            
            print(sklearn.metrics.accuracy_score(outcome_data['target'].tolist(),
                                              outcome_data['prediction'].tolist()))

    # Relabel indices for feature and target data
    record_count = int(feature_data.shape[0])
    
    # Ensure that we have enough records
    if record_count < min_record_count:
        continue

    '''
    If we have at least that many records, let's actually 
    train a model.
    '''

    feature_data.index = range(record_count)
    target_data.index = range(record_count)  

    # Subset feature labels to exclude our indices
    if num_dockets > min_record_count and model == None:
        # Set the excluded feature labels
        feature_labels = [label for label in feature_data.columns.tolist() if label not in bad_feature_labels]

        # Train the model on the data
        model = train_model(feature_data[feature_labels],  target_data[0].apply(int).tolist(),
                            search_parameters)
        
    elif num_dockets > min_record_count and num_dockets % 100 == 0:
        print((docket_id, num_dockets))

        # Train the model on the data
        model = train_model(feature_data[feature_labels],  target_data[0].apply(int).tolist(),
                            search_parameters)

# Track the case assessment
case_assessment = []

# Try to calculate case outcomes accurately.
for case_id, case_data in outcome_data.groupby('docket'):
    # Get the vote data
    vote_data = (case_data[['docket', 'justice', 'is_chief', 'justice_direction_mean', 'prediction', 'target']].sort('justice_direction_mean'))
    overturn_predicted = vote_data['prediction'].mean()
    overturn_actual = vote_data['target'].mean()
    row = [case_id,
                           get_year_from_docket(case_id),
                            case_data['issue'].tail(1).tolist().pop(),
                            case_data['issue_area'].tail(1).tolist().pop(),
                            case_data['case_source_circuit'].tail(1).tolist().pop(),
                            case_data['case_origin_circuit'].tail(1).tolist().pop(),
                            case_data['lc_direction'].tail(1).tolist().pop(),
                            case_data['lc_disposition'].tail(1).tolist().pop(),
                            overturn_predicted,
                            overturn_actual,
                            overturn_predicted > 0.5,
                            overturn_actual > 0.5]
    
    # Get the votes aligned
    for value in vote_data['prediction']:
        row.append(value)
        
    # Pad if fewer than nine justices voting
    if vote_data['prediction'].shape[0] < 9:
        for i in range((9 - vote_data['prediction'].shape[0])):
            row.append(numpy.nan)
    
    row.append(vote_data.ix[vote_data['is_chief'] == 1]['prediction'].tolist().pop())

    # Append to the case assessment dataframe.
    case_assessment.append(row)

# Setup the column list and final case assessment DF
column_list = ['docket', 'year', 'issue', 'issue_area', 'case_source_circuit',
               'case_origin_circuit', 'lc_direction', 'lc_disposition',
               'overturn_count_predict', 'overturn_count_actual', 'overturn_predict',
               'overturn_actual', 'justice_1', 'justice_2', 'justice_3', 'justice_4',
               'justice_5', 'justice_6', 'justice_7', 'justice_8', 'justice_9',
               'justice_chief']
    
case_assessment_df = pandas.DataFrame(case_assessment, columns=column_list)
case_assessment_df['correct'] = (case_assessment_df['overturn_predict'] == case_assessment_df['overturn_actual'])
        
outcome_data['correct'] = (outcome_data['prediction'] == outcome_data['target'])

# Get the annual accuracy figures
outcome_data['year'] = outcome_data['docket'].apply(get_year_from_docket)
case_assessment_df['year'] = case_assessment_df['docket'].apply(get_year_from_docket)

x_case_assessment_df = case_assessment_df.ix[case_assessment_df['year'] >= 1946]

print("Case Assessment")
print(pandas.DataFrame(sklearn.metrics.confusion_matrix(x_case_assessment_df['overturn_actual'].tolist(),
                                              x_case_assessment_df['overturn_predict'].tolist())))

print(sklearn.metrics.classification_report(x_case_assessment_df['overturn_actual'].tolist(),
                                               x_case_assessment_df['overturn_predict'].tolist()))

print(sklearn.metrics.accuracy_score(x_case_assessment_df['overturn_actual'].tolist(),
                                              x_case_assessment_df['overturn_predict'].tolist()))

print("Justice Assessment")
x_outcome_data = outcome_data.loc[outcome_data['year'] >= 1946]

print(pandas.DataFrame(sklearn.metrics.confusion_matrix(x_outcome_data['target'].tolist(),
                                              x_outcome_data['prediction'].tolist())))

print(sklearn.metrics.classification_report(x_outcome_data['target'].tolist(),
                                              x_outcome_data['prediction'].tolist()))

print(sklearn.metrics.accuracy_score(x_outcome_data['target'].tolist(),
                                              x_outcome_data['prediction'].tolist()))

# Setup vars
output_folder = 'model_output'
timestamp_suffix = time.strftime("%Y%m%d%H%M%S")

# Create path
run_output_folder = os.path.join(output_folder, timestamp_suffix)
os.makedirs(run_output_folder)

# Output data
outcome_data.to_csv(os.path.join(run_output_folder, 'justice_outcome_data.csv'))
case_assessment_df.to_csv(os.path.join(run_output_folder, 'case_outcome_data.csv'))

# Make a ZIP
os.system('zip -9 {0}.zip {1}'.format(os.path.join(output_folder, timestamp_suffix),
                                      os.path.join(run_output_folder, '*.csv')))

