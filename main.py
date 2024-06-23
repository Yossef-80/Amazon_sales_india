# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import streamlit as st



import geopandas as gpd
import folium
from geopy import Photon
from geopy.exc import GeocoderTimedOut


# STEP 1
# 1.1  Data inspection
def load_dataset():
    # 1.1.1 (load data)
    data = pd.read_csv("Amazon Sale Report.csv", low_memory=False)

    # 1.1.2 (inspect first 5 rows)
    print(data.head())

    # 1.1.3 (check data types of columns and issues)
    print(data.info())

    # the issues with the data type of non-numerical columns, as it is identified as object not string
    return data


# 1.2 Summary statistics
def summary_stats(data):
    # 1.2.1 (generate summary statistics)
    print(data.describe())

    categorical_columns = data.select_dtypes(include=['object']).columns
    # print(categorical_columns)
    for category in categorical_columns:
        print(data[category].value_counts())

    # 1.2.2 (visualization of distribution of key features)
    sns.set_style("darkgrid")
    # =>between the clothes sizes and the amount
    # sns.barplot(x='Size', y='Amount', data=data)
    # plt.show()
    # =>between the state and the amount
    # sns.barplot(x='ship-state', y='Amount', data=data)
    # plt.xticks(rotation=90)
    # plt.show()
    # plt.imshow(data, cmap='hot', interpolation='nearest')
    # plt.show()

    # 1.2.3 key features are
    # (Date,Status,Fulfilment,ship-service-level,Category,Size,Courier Status,Qty,Amount,ship-city,ship-state,B2B)


# STEP 2 DATA PROCESSING

# 2.1 handling missing values
def handle_missing(data):
    # print(data.isna())

    print(data.isna().sum())
    new_data = data.iloc[:, 0:-1]

    # the sales in india so the currency will be all in "INR"
    new_data.fillna({'currency': 'INR'}, inplace=True)
    # when the "Qty" is 0 the "Courier Status" is canceled
    new_data.loc[new_data['Qty'] == 0, 'Courier Status'] = 'Cancelled'
    # the amount is 0 if the Qty=0 or the status is cancelled
    new_data.loc[new_data['Amount'].isna(), 'Amount'] = 0
    # drop the 33 ship country=>then it removes the 33 NA ship-city , ship-state and ship-postal-code
    new_data.dropna(subset=['ship-country'], inplace=True)
    # fill empty "fulfilled-by" with 'Non-easy Ship'
    new_data.fillna({'fulfilled-by': 'Non-easy Ship'}, inplace=True)

    # fill empty "promotion-id" with 'No promotion' as no promotion applied
    new_data.fillna({'promotion-ids': 'No promotion'}, inplace=True)

    return new_data


def convert_data_type(data):
    print(data.dtypes)
    print(data.columns)
    # no need for index or order ID column
    data = data.iloc[:, 2:]
    # print(data.head())
    # convert date
    data['Date'] = pd.to_datetime(data['Date'], format='%m-%d-%y', errors='coerce')
    # convert Status ,Fulfilment,Sales channel,ship-service-level,Style,Category,Size,Courier Status to categorical ASIN,SKU to string
    data['Status'] = pd.Categorical(data.Status)
    data['Fulfilment'] = pd.Categorical(data.Fulfilment)
    data['Sales Channel '] = data['Sales Channel '].astype('category')
    data['ship-service-level'] = data['ship-service-level'].astype('category')
    data['Style'] = pd.Categorical(data.Style)
    data['ASIN'] = data['ASIN'].astype(str)
    data['SKU'] = data['SKU'].astype(str)
    # convert Category ,Size,Courier Status,currency,ship-country,ship-state  ,ship-city,promotion-ids,fulfilled-by  to categorical
    data['Category'] = pd.Categorical(data.Category)
    data['Size'] = pd.Categorical(data.Size)
    data['Courier Status'] = data['Courier Status'].astype('category')
    data['currency'] = pd.Categorical(data.currency)
    data['ship-country'] = data['ship-country'].astype(str)
    data['ship-state'] = data['ship-state'].astype(str)
    data['ship-city'] = data['ship-city'].astype(str)
    data['promotion-ids'] = data['promotion-ids'].astype('category')
    data['fulfilled-by'] = data['fulfilled-by'].astype('category')
    print(data.dtypes)

    return data


# 2.3 Outlier detection and Treatment
def outliers(data):
    sns.boxplot(data['Amount'])
    plt.show()
    sns.boxplot(data['Qty'])
    plt.show()

    # Amount IQR
    print("old Shape: ", data.shape)
    print(data['Amount'].describe())

    Q1 = data['Amount'].quantile(0.25)
    Q3 = data['Amount'].quantile(0.75)
    print("Q1", Q1, "Q3", Q3)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    data = data[(data['Amount'] > lower) & (data['Amount'] < upper)]
    sns.boxplot(data['Amount'])
    plt.show()
    print("New Shape: ", data.shape)
    print(data['Amount'].describe())

    # Qty

    print(data['Qty'].value_counts())
    # there is only 237 record has value greater than 1 and the remaining values are 0 or 1
    data = data[(data['Qty']) <= 1]
    print(data['Qty'].value_counts())
    return data


# STEP 3 (Data Visualization)
def visualize_matplotlib_seaborn(data):
    # Histogram of Amount
    plt.hist(data['Amount'])
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.show()
    # barplots

    sns.barplot(data=data, y='Amount', x='Category')
    plt.title('mean of Amounts by Category')

    plt.show()

    sns.barplot(data=data, y='Amount', x='Size')
    plt.title('mean of Amounts by Size')
    plt.show()
    # line plot
    sns.lineplot(data=data, x='Date', y='Amount')
    plt.show()

    # heatmaps
    df_heatmap = data.pivot_table(values='Amount', index='ship-state', columns='Category', aggfunc=np.mean)

    plt.figure(figsize=(15, 10))

    print(df_heatmap)
    sns.heatmap(df_heatmap, cmap='YlGnBu', linewidths=0.6)
    plt.show()

    df_heatmap = data.pivot_table(values='Amount', index='ship-state', columns='Size', aggfunc=np.mean)

    plt.figure(figsize=(15, 10))

    print(df_heatmap)
    sns.heatmap(df_heatmap, cmap='YlGnBu', linewidths=0.6)
    plt.show()


def visual_analysis(data):
    # Visualize sales trends over time
    sns.lineplot(data=data, x='Date', y='Amount')
    plt.show()
    # number of categories sold over time
    aggregated_data = data.groupby(['Date', 'Category']).size().reset_index(name='Count')

    sns.lineplot(data=aggregated_data, x='Date', y='Count', hue='Category', marker='o')
    plt.show()
    # number of sizes sold over time
    aggregated_data = data.groupby(['Date', 'Size']).size().reset_index(name='Count')

    sns.lineplot(data=aggregated_data, x='Date', y='Count', hue='Size', )
    plt.show()

    # Top-selling products and categories
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='Category', order=data['Category'].value_counts().index)
    plt.title('Count of Each Category sold')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.show()
    '''
    category_summary = data.groupby('ship-state')['Amount'].sum().reset_index()

    geo_dataframe = pd.DataFrame({
        'ship-state': category_summary['ship-state'].tolist(),
        'Total_amount': category_summary['Amount'].tolist()
    })
    geolocator = Photon(user_agent="geoapiExercises", timeout=10)

    # Function to geocode with retry
    def geocode_with_retry(city):
        try:
            return geolocator.geocode(city)
        except GeocoderTimedOut:
            return geocode_with_retry(city)

    # Geocode the cities
    geo_dataframe['Coordinates'] = geo_dataframe['ship-state'].apply(geocode_with_retry)

    # Extract latitude and longitude
    geo_dataframe['Latitude'] = geo_dataframe['Coordinates'].apply(lambda x: x.latitude if x else None)
    geo_dataframe['Longitude'] = geo_dataframe['Coordinates'].apply(lambda x: x.longitude if x else None)
    print(geo_dataframe['Latitude'])
    print(geo_dataframe['Longitude'])
    geo_dataframe = geo_dataframe.dropna(subset=['Latitude', 'Longitude'])
    fig = px.scatter_mapbox(
        geo_dataframe,
        lat="Latitude",
        lon="Longitude",
        text="ship-state",
        size="Total_amount",
        color_discrete_sequence=["yellow"],
        # color="Total_amount",
        # color_continuous_scale=px.colors.cyclical.IceFire,
        size_max=100,
        zoom=3,
        mapbox_style="open-street-map",
        title="City Amounts Map"
    )
    fig.update_traces(
        textfont=dict(color='black')

    )
    # Use geo_dataframe to get center coordinates
    fig.update_layout(
        mapbox=dict(
            center={"lat": geo_dataframe['Latitude'].mean(), "lon": geo_dataframe['Longitude'].mean()},
            zoom=5
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    fig.show()
    '''


# Step 4 Predictive Modeling

# 4.1 Building Predictive Models
def predictive_models(data):
    logr = LogisticRegression(max_iter=500)
    # encoding Data
    encoder = OneHotEncoder(sparse_output=False)
    label_encoder = LabelEncoder()
    scaler = StandardScaler()

    # newData=data[['Status','Fulfilment','Sales Channel ','ship-service-level','Style','Category','Size',
    # 'Courier Status', 'Qty','Amount','ship-city','ship-state','ship-country','fulfilled-by',
    # 'B2B']] categorical_columns = newData.select_dtypes(include=['object','category']).columns.tolist()
    categorical_columns = [ 'Category', 'Size',
                           'ship-state','fulfilled-by']
    print(categorical_columns)
    # data=data[categorical_columns]
    unique_counts = data.apply(lambda x: x.nunique())
    data['Status_encoded'] = label_encoder.fit_transform(data['Status'])
    data['ship-service-level_encoded']=label_encoder.fit_transform(data['ship-service-level'])
    data['Fulfilment_encoded']=label_encoder.fit_transform(data['Fulfilment'])
    data['Sales Channel_encoded']=label_encoder.fit_transform(data['Sales Channel '])
    data['B2B_encoded']=label_encoder.fit_transform(data['B2B'])
    data['Courier Status_encoded']=label_encoder.fit_transform(data['Courier Status'])
    print(unique_counts)
    newData = data[categorical_columns]
    one_hot_encoded = encoder.fit_transform(newData)
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    df_encoded = pd.concat([data['Status_encoded'],data['ship-service-level_encoded'],data['Fulfilment_encoded'],data['Sales Channel_encoded'],data['B2B_encoded'],data['Courier Status_encoded'], data['Amount'], data['Qty'], one_hot_df], axis=1)
    # df_encoded = df_encoded.drop(categorical_columns, axis=1)
    print(df_encoded.isna().sum())
    print(df_encoded.isnull().sum())
    df_encoded.dropna(inplace=True)
    print(f"Encoded Employee data : \n{df_encoded}")
    #df_encoded=df_encoded[df_encoded['Status_encoded'].isin([0, 3, 5])]
    numerical_columns = ['Qty', 'Amount']
    X_data = df_encoded.iloc[:, 1:]
    X_data[numerical_columns] = scaler.fit_transform(X_data[numerical_columns])

    print(X_data.head())
    Y_data = df_encoded['Status_encoded']
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, random_state=100, train_size=0.8)
    print(x_train.shape)
    print(x_test.shape)
    logr.fit(x_train, y_train)
    y_pred = logr.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted',zero_division=0)
    recall = recall_score(y_test, y_pred,average='weighted',zero_division=0)
    f1 = f1_score(y_test, y_pred,average='weighted',zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred,zero_division=0)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nClassification Report:")
    print(class_report)
    #Decision Tree Classifier

    clf_gini = DecisionTreeClassifier(criterion="gini",  random_state=100, max_depth=8, min_samples_leaf=7)
    clf_gini.fit(x_train, y_train)
    y_pred = clf_gini.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='weighted',zero_division=0)
    recall = recall_score(y_test, y_pred,average='weighted',zero_division=0)
    f1 = f1_score(y_test, y_pred,average='weighted',zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred,zero_division=0)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nClassification Report:")
    print(class_report)

    # Random Forest
    clf = RandomForestClassifier(n_estimators=250)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nClassification Report:")
    print(class_report)

    #Cross Validation
    k_folds = KFold(n_splits=5)
    scores = cross_val_score(clf_gini, X_data, Y_data, cv=k_folds)
    print("Cross Validation Scores: ", scores)
    print("Average CV Score: ", scores.mean())
    print("Number of CV Scores used in Average: ", len(scores))


def export_data_set(data):
    data.to_csv(r"E:\slash task\python task\AmazonSalesReport_cleaned.csv", index=False)

if __name__ == '__main__':
    data = load_dataset()
    summary_stats(data)
    clean_data = handle_missing(data)

    print(clean_data.isna().sum())
    converted_data = convert_data_type(clean_data)
    clean_data_2 = outliers(converted_data)
    visualize_matplotlib_seaborn(clean_data_2)
    visual_analysis(clean_data_2)
    predictive_models(clean_data_2)
    #export_data_set(clean_data_2)