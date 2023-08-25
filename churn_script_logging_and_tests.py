'''
Module containing tests and logging for functions for churn study (in churn_library.py)

Author : Krystelle Lionti

Date : August 2023
'''

import os
import logging
import pytest
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def df_plugin():
    """
    Create pystest Namespace
    """
    return None

def pytest_configure():
    """
    Create Dataframe object in Namespace
    """
    pytest.df = df_plugin()

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: SUCCESS - The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: ERROR - The file doesn't appear to have rows and columns")
        raise err
    pytest.df = data_frame


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    # Create DataFrame
    data_frame = pytest.df

    # Call perform_eda function
    cls.perform_eda(data_frame)

    plot_list = ['churn_plot', 'Total_Trans_Ct_plot', 'heatmap']
    for plot in plot_list:
        try:
            # Check if the histogram plot file exists
            assert os.path.exists('./images/eda/'+plot+'.png')

            logging.info("Testing perform_eda: SUCCESS, found %s", plot)
        except AssertionError as err:
            logging.error("Testing perform_eda: ERROR - The path for saving %s was NOT FOUND", plot)
            raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    # Create DataFrame
    data_frame = pytest.df

    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # Call encoder_helper function
    data_frame = cls.encoder_helper(data_frame, category_lst)

    # check if the Churn column exists and is of the correct datatype int64
    try:
        assert data_frame['Churn'].dtype == 'int64'
        logging.info("Testing encoder_helper: SUCCESS - Churn column successfully created and of the right datatype 'int64'")
    except AssertionError:
        logging.error("Testing encoder_helper: ERROR - datatype of Churn column is not the expected 'int64'")
    except KeyError:
        logging.error("Testing encoder_helper: ERROR - no Churn column found")

    # Check if the new encoded columns exist and are of the correct datatype 'float'
    for column in category_lst:
        encoded_column_name = f"{column}_Churn"
        try:
            assert data_frame[encoded_column_name].dtype == 'float'
            logging.info("Testing encoder_helper: SUCCESS - %s column successfully created and of the right datatype 'float'", encoded_column_name)
        except AssertionError:
            logging.error("Testing encoder_helper: ERROR - datatype of %s column is not the expected 'float'", encoded_column_name)
        except KeyError:
            logging.error("Testing encoder_helper: ERROR - no %s column found", encoded_column_name)

    pytest.dataframe = data_frame


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn',
        'Churn']

    # Create DataFrame
    data_frame = pytest.df

    # Call encoder_helper function
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(data_frame)

    # Check if the only columns remaining are those listed in keep_cols for X_train, X_test, y_train, y_test
    try:
        assert set(X_train.columns) == set(keep_cols[:-1])  # Exclude 'Churn' column
        logging.info("Testing perform_feature_engineering: SUCCESS - Only the expected columns are remaining in X_train")
    except AssertionError:
        logging.error("Testing perform_feature_engineering: ERROR - Unexpected columns found in X_train")

    try:
        assert set(X_test.columns) == set(keep_cols[:-1])  # Exclude 'Churn' column
        logging.info("Testing perform_feature_engineering: SUCCESS - Only the expected columns are remaining in X_test")
    except AssertionError:
        logging.error("Testing perform_feature_engineering: ERROR - Unexpected columns found in X_test")

    try:
        assert y_train.name == keep_cols[-1]  # 'Churn' column
        logging.info("Testing perform_feature_engineering: SUCCESS - y_train column name is Churn as expected")
    except AssertionError:
        logging.error("Testing perform_feature_engineering: ERROR - Unexpected column name for y_train")

    try:
        assert y_test.name == keep_cols[-1]  # 'Churn' column
        logging.info("Testing perform_feature_engineering: SUCCESS - y_test column name is Churn as expected")
    except AssertionError:
        logging.error("Testing perform_feature_engineering: ERROR - Unexpected column name for y_test")

    try:
        # Check if the shapes of the train and test sets are as expected
        assert X_train.shape[0] == int(0.7 * data_frame.shape[0])
        assert X_test.shape[0] == data_frame.shape[0] - int(0.7 * data_frame.shape[0])
        assert y_train.shape[0] == int(0.7 * data_frame.shape[0])
        assert y_test.shape[0] == data_frame.shape[0] - int(0.7 * data_frame.shape[0])

        # Check if the train and test sets have the correct number of features
        assert X_train.shape[1] == len(keep_cols) - 1  # Exclude 'Churn' column
        assert X_test.shape[1] == len(keep_cols) - 1  # Exclude 'Churn' column

        logging.info("Testing perform_feature_engineering: SUCCESS - Train-test split test passed successfully!")

    except AssertionError:
        logging.error("Testing perform_feature_engineering: ERROR - Train-test split test failed!")
        logging.info("df.shape[0]: %d", data_frame.shape[0])
        logging.info("X_train.shape[0]: %d, y_train.shape[0]: %d", X_train.shape[0], y_train.shape[0])
        logging.info("X_test.shape[0]: %d, y_test.shape[0]: %d", X_test.shape[0], y_test.shape[0])

    pytest.X_train = X_train
    pytest.X_test = X_test
    pytest.y_train = y_train
    pytest.y_test = y_test

def test_train_models(train_models):
    '''
    test train_models
    '''
    # Create test and train sets
    X_train = pytest.X_train
    X_test = pytest.X_test
    y_train = pytest.y_train
    y_test = pytest.y_test

    # Call train_models
    cls.train_models(X_train, X_test, y_train, y_test)

    try:
        # Check if the models exist
        assert os.path.exists('./models/random_forest_model.pkl')
        assert os.path.exists('./models/logistic_regression_model.pkl')
        logging.info("Testing train_models: SUCCESS - Models were created and saved successfully.")
    except AssertionError:
        logging.error("Testing train_models: ERROR - Models were not created and/or saved.")

    try:
        # Check if the reports exist
        assert os.path.exists('./images/results/generated_lr_report.png')
        assert os.path.exists('./images/results/generated_rf_report.png')
        logging.info("Testing train_models: SUCCESS - Reports were created and saved successfully.")
    except AssertionError:
        logging.error("Testing train_models: ERROR - Reports were not created and/or saved.")

    try:
        # Check if the ROC curves exist
        assert os.path.exists('./images/results/roc_curve_lr.png')
        assert os.path.exists('./images/results/roc_curve_rf.png')
        logging.info("Testing train_models: SUCCESS - ROC curves were created and saved successfully.")
    except AssertionError:
        logging.error("Testing train_models: ERROR - ROC curves were not created and/or saved.")


if __name__ == "__main__":
    # test_import("./data/bank_data.csv")
    test_eda(test_import("./data/bank_data.csv"))
    # test_encoder_helper(test_import("./data/bank_data.csv"))
    # test_perform_feature_engineering(test_encoder_helper(test_import("./data/bank_data.csv")))
    test_train_models(test_perform_feature_engineering(test_encoder_helper(test_import("./data/bank_data.csv"))))
