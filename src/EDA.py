from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from yellowbrick.classifier import ConfusionMatrix
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ROCAUC
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import DiscriminationThreshold
from sklearn.linear_model import RidgeClassifier
from yellowbrick.classifier import PrecisionRecallCurve

# in your notebook cell
##import sys

# path relative to your notebook
#sys.path.insert(0, '../src')

# import as usual
import fun_functions as fun


def read_csv(filepath):

    return pd.read_csv(filepath)


def clean_world_flights():
    dataframe = read_csv('../data/API_IS.AIR.DPRT_DS2_en_csv_v2_5735762/API_IS.AIR.DPRT_DS2_en_csv_v2_5735762.csv')
    dataframe = dataframe[dataframe['Country Name'] == 'World']
    dataframe = dataframe.drop(columns=['Country Name'
                                        ,'Country Code'
                                        ,'Indicator Name'
                                        ,'Indicator Code'])
    dataframe = dataframe.swapaxes(axis1=0, axis2=1, copy=None).reset_index()
    dataframe.rename(columns={'index': 'Year', 259: 'Flights'}, inplace=True)
    dataframe.fillna(0, inplace=True)
    dataframe = dataframe[dataframe['Flights'] > 0]
    df2 = pd.DataFrame({'Year': [1972, 2022], 'Flights': [9634700.00, 32400000]})
    dataframe = pd.concat([dataframe, df2],ignore_index=True)
    dataframe['Year'] = dataframe['Year'].astype(int)
    dataframe = dataframe.sort_values(by='Year', ascending=True).reset_index()
    dataframe = dataframe[dataframe['Year'] >= 2000]
    return dataframe

def total_flights_per_year():
    df = clean_world_flights()

    fig, ax = plt.subplots(figsize = (16, 9))
    sns.lineplot(df, y='Flights', x='Year', linewidth = 3)
    ax.legend()
    ax.set_ylabel('Flights in Tens of Millions', fontsize=15)
    ax.set_xlabel('Year', fontsize=15)
    ax.set_title('Total Flights', style= 'italic', fontsize=18)

def incidents_by_year():
    dataframe = read_csv('../data/Aircraft_Incident_Dataset1.csv')
    dataframe['Incident_Date'] = pd.to_datetime(dataframe['Incident_Date'], format="%m/%d/%Y")
    dataframe['Year'] = dataframe['Incident_Date'].dt.year
    dataframe = dataframe[dataframe['Year'] >= 2000]
    dataframe['incident_count'] = 1
    dataframe1 = dataframe.groupby('Year')['incident_count'].sum().reset_index()
    ave_median_incident = dataframe1['incident_count'].median()

    fig, ax = plt.subplots(figsize = (16, 9))
    sns.lineplot(dataframe1, y='incident_count', x='Year', color= 'b', linewidth = 3)
    ax.axhline(ave_median_incident, color = 'r', label=f'Average Incidents: {ave_median_incident:0.0f}', linestyle='--')
    ax.legend()
    ax.set_ylabel('Incidents', fontsize=15)
    ax.set_title('Total Incidents', style= 'italic', fontsize=18)
    ax.set_xlabel('Year', fontsize=15)
    plt.show()
    return dataframe1


def occupants(df, col, index):
    onboard_split = []
    for x in df[col]:
        onboard_split.append(x.split('/'))

    occupants = []

    for list in onboard_split:
        occupants.append(list[index])

    list_occupants = []

    for list in occupants:
        list_occupants.append(list.split(':'))

    num_occupants = []

    for item in list_occupants:
        if item[1].isspace() == True:
            num_occupants.append(0)
        elif item[1] == '':
            num_occupants.append(0)
        else:
            num_occupants.append(int(item[1]))


    return num_occupants



def clean_main_df():
    df = read_csv('../data/Aircraft_Incident_Dataset1.csv')
    df['Incident_Date'] = pd.to_datetime(df['Incident_Date'], format="%m/%d/%Y")
    df['Year'] = df['Incident_Date'].dt.year
    df = df[df['Year'] >= 2000]
    df['incident_count'] = 1
    df['Total Occupants'] = occupants(df, 'Onboard_Total', 1)
    df['Total Passengers'] = occupants(df, 'Onboard_Passengers', 1)
    df['Total Crew'] = occupants(df, 'Onboard_Crew', 1)
    df['Total Onboard Fatalities'] = occupants(df, 'Onboard_Total', 0)
    df['Total Passenger Fatalities'] = occupants(df, 'Onboard_Passengers', 0)
    df['Total Crew Fatalities'] = occupants(df, 'Onboard_Crew', 0)
    return df

def incidents_by_phase():
    df = clean_main_df()
    df = pd.DataFrame(df['Aircraft_Phase'].value_counts()).reset_index()
    
    fig, ax = plt.subplots()
    sns.barplot(df, y='Aircraft_Phase', x='count')
    ax.set_ylabel('Aircraft Phase')
    ax.set_xlabel('Total Incidents')
    ax.set_title('Total Incidents by Aircraft Phase\n2000-2022', style= 'italic')
    plt.show()

def incidents_by_category():
    df = clean_main_df()
    df = df[['Year', 'incident_count', 'Incident_Category', 'Total Onboard Fatalities']]
    df = df.groupby('Incident_Category').sum().reset_index()
    df = df.sort_values(by='incident_count', ascending=False)

    sns.barplot(df, y='Incident_Category', x='incident_count')

    plt.show()
    return df


def top_incident_categories_by_fatalities():
    df = incidents_by_category()

    df = df[df['Total Onboard Fatalities'] >50]
    df = df.sort_values(by='Total Onboard Fatalities', ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(df, y='Incident_Category', x='Total Onboard Fatalities')
    ax.set_ylabel('Incident Category')
    ax.set_title('Top Incident Categories by Occupant Fatalities\n2000-2022', style= 'italic')

    plt.show()

def fatal_or_not():
    df = clean_main_df()
    fatal_or_not = []

    for fatal in df['Total Onboard Fatalities']:
        if fatal > 0:
            fatal_or_not.append(1)
        else:
            fatal_or_not.append(0)
    df['Fatal or Not'] = fatal_or_not
    return df


def chi2_test():
    df = fatal_or_not()
    # Here, we are focusing on one specific column 'General_Health' and 'Heart_Disease'
    cross_tab = pd.crosstab(df['Fatal or Not'], df['Incident_Cause(es)'])
    # Perform the chi-square test
    chi2_stat, p_val, dof, expected = chi2_contingency(cross_tab)
    # Output the results
    print("Chi-Square Statistic:", chi2_stat)
    print("P-value:", p_val)
    print("Degrees of Freedom:", dof)
    print("Expected Frequencies Table:")
    print(pd.DataFrame(expected, index=cross_tab.index, columns=cross_tab.columns))
    # Interpret the results
    alpha = 0.05
    print("\nSignificance Level (alpha):", alpha)
    print("Is the p-value less than alpha?", p_val < alpha)
    if p_val < alpha:
        print("Reject the Null Hypothesis (H0): There is a significant association between Incident Causes and Fatal Incidents.")
    else:
        print("Fail to Reject the Null Hypothesis (H0): There is no significant association between Incident Causes and Fatal Incidents.")



def log_regression_test_on_causes():
    df = fatal_or_not()
    causes = ['']

    for cause in df['Incident_Cause(es)']:
        new = cause.split('-')
        causes.append(new[1])
    
    causes.remove(causes[0])

    df['Incident Cause'] = causes

    df = df.drop(columns=['Aircaft_Registration'
                            ,'Year', 'Total Passenger Fatalities', 'Total Crew Fatalities'
                            , 'Total Crew', 'Total Passengers', 'Incident_Date'
                            , 'Incident_Location', 'Ground_Casualties'
                            ,'Date', 'Time', 'Arit'
                            ,'Onboard_Crew', 'Onboard_Total', 'Fatalities'
                            ,'Onboard_Passengers', 'Aircaft_First_Flight', 'Collision_Casualties'
                            ,'Departure_Airport', 'Destination_Airport', 'incident_count'
                            ,'Total Occupants', 'Total Onboard Fatalities'])
    y = df['Fatal or Not']
    X = pd.get_dummies(df, columns=['Incident Cause'],prefix='',prefix_sep='', dtype=int)
    X = X.drop(columns=['Aircaft_Model'
                        , 'Aircaft_Nature'
                        ,'Incident_Category'
                        ,'Aircaft_Operator'
                        ,'Aircaft_Damage_Type'
                        , 'Aircaft_Engines'
                        , 'Aircraft_Phase'
                        , 'Fatal or Not'
                        , 'Incident_Cause(es)'])
    X_train, X_test, y_train, y_test = fun.train_test_split(X, y, test_size=.3, random_state=40)

    model = fun.LogisticRegressionCV(cv=10, random_state=40, max_iter=10000)
    # Fit model
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def confusion_matrix():
    model, X_train, X_test, y_train, y_test = log_regression_test_on_causes()

    cm = ConfusionMatrix(model)
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)

    cm.show()


def ROC_curve_for_model():
    model, X_train, X_test, y_train, y_test = log_regression_test_on_causes()

    visualizer = ROCAUC(model)

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    visualizer.show() 


def plot_coefficients():
    model, X_train, X_test, y_train, y_test = log_regression_test_on_causes()

    model_output1 = pd.DataFrame()
    model_output1['Coeficient'] = (model.coef_).flatten()
    model_output1['Feature Name'] = model.feature_names_in_
    model_output1 = model_output1.sort_values(by='Coeficient', ascending=False)

    new_features = []
    for line in model_output1['Feature Name']:
        if ', Result' in line:
            new_line = line.replace(', Result ', '')

            new_features.append('Result of' + new_line)
        else:
            new_features.append(line)

    model_output1['Feature Name'] = new_features

    fig, ax = plt.subplots(figsize= (16, 50))
    sns.barplot(model_output1, y='Feature Name', x='Coeficient')


def random_forest_on_data():
    model, X_train, X_test, y_train, y_test = log_regression_test_on_causes()
    classes = ['Non-Fatal', 'Fatal']
    visualizer = ClassPredictionError(
        RandomForestClassifier(random_state=42, n_estimators=10), classes=classes
    )

    # Fit the training data to the visualizer
    visualizer.fit(X_train, y_train)

    # Evaluate the model on the test data
    visualizer.score(X_test, y_test)

    # Draw visualization
    visualizer.show()


def find_discrimination_threshold():
    model, X_train, X_test, y_train, y_test = log_regression_test_on_causes()
    model = LogisticRegression(multi_class="auto", solver="liblinear")
    visualizer = DiscriminationThreshold(model)

    visualizer.fit(X_train, y_train)        # Fit the data to the visualizer
    visualizer.show() 

def precision_recall_curve():
    model, X_train, X_test, y_train, y_test = log_regression_test_on_causes()
    viz = PrecisionRecallCurve(RidgeClassifier(random_state=0))
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show()


