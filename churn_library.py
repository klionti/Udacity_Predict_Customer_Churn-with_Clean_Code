'''
Module containing functions for churn study

Author : Krystelle Lionti

Date : August 2023
'''

# import libraries
import os
import seaborn as sns
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, RocCurveDisplay
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    data_frame = pd.read_csv(pth)
    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    # Create and save churn histogram (univariate, categorical plot)
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    data_frame['Churn'].hist()
    plt.savefig('./images/eda/churn_plot.png')

    # Create and save total_transaction_count histogram (univariate,
    # quantitative plot)
    plt.figure(figsize=(20, 10))
    sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/Total_Trans_Ct_plot.png')

    # Create and save heatmap (bivariate plot)
    numeric_columns = data_frame.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame[numeric_columns].corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap.png')


def encoder_helper(data_frame, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            data_frame: pandas dataframe with new columns
    '''

    # Create Churn column
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Encode all columns listed in category_lst
    for column in category_lst:
        encoded_lst = []
        category_means = {}

        for category in data_frame[column].unique():
            mean_churn = data_frame[data_frame[column] == category]['Churn'].mean()
            category_means[category] = mean_churn

        for val in data_frame[column]:
            encoded_lst.append(category_means[val])

        encoded_column_name = f"{column}_Churn"
        data_frame[encoded_column_name] = encoded_lst
    return data_frame

def perform_feature_engineering(data_frame):
    '''
    input:
              data_frame: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
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

    clean_df = data_frame[keep_cols]
    X = clean_df.drop('Churn', axis=1)
    y = clean_df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # Generate classification report for training data
    train_report_lr = classification_report(y_train, y_train_preds_lr)
    train_report_rf = classification_report(y_train, y_train_preds_rf)

    # Generate classification report for testing data
    test_report_lr = classification_report(y_test, y_test_preds_lr)
    test_report_rf = classification_report(y_test, y_test_preds_rf)

    # Save LR classification reports as a single image
    combined_lr_report = f"Train Report (LR):\n\n{train_report_lr}\n\n" \
                     f"Test Report (LR):\n\n{test_report_lr}"
    # Create an image with white background
    image_width = 800
    image_height = 600
    background_color = (255, 255, 255)
    image = Image.new('RGB', (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)

    # Set font properties
    font_size = 20
    font_color = (0, 0, 0)
    font = ImageFont.truetype('arial.ttf', font_size)

    # Write the combined_lr_report text on the image
    text_position = (50, 50)
    draw.text(text_position, combined_lr_report, font=font, fill=font_color)

    # Save the image
    image.save('./images/results/generated_lr_report.png')


    # Save RF classification reports as a single image
    combined_rf_report = f"Train Report (RF):\n\n{train_report_rf}\n\n" \
                        f"Test Report (RF):\n\n{test_report_rf}"
    # Create an image with white background
    image_width = 800
    image_height = 600
    background_color = (255, 255, 255)
    image = Image.new('RGB', (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)

    # Set font properties
    font_size = 20
    font_color = (0, 0, 0)
    font = ImageFont.truetype('arial.ttf', font_size)

    # Write the combined_lr_report text on the image
    text_position = (50, 50)
    draw.text(text_position, combined_rf_report, font=font, fill=font_color)

    # Save the image
    image.save('./images/results/generated_rf_report.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Check if the target variable is present in X_data
    if 'Churn' in X_data.columns:
        # Remove the target variable from X_data
        X_data = X_data.drop('Churn', axis=1)

    # Get the feature importances from the model
    importances = model.feature_importances_

    # Get the feature names from X_data
    feature_names = X_data.columns

    # Sort the feature importances and feature names together
    sorted_indices = importances.argsort()[::-1]
    importances = importances[sorted_indices]
    feature_names = feature_names[sorted_indices]

    # Create a horizontal bar plot of feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances, align='center')
    plt.yticks(range(len(importances)), feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance Plot')
    plt.tight_layout()

    # Specify the desired filename and extension
    filename = "feature_importance_plot.png"
    
    # Save the plot to the specified output path
    plt.savefig(output_pth + filename)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              y_train_preds_lr: training predictions from logistic regression
              y_train_preds_rf: training predictions from random forest
              y_test_preds_lr: test predictions from logistic regression
              y_test_preds_rf: test predictions from random forest
              cv_rfc.best_estimator_: random forest trained model
              lrc: logistic regression trained model
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    lrc.fit(X_train, y_train)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Generate classification report for training data
    train_report_lr = classification_report(y_train, y_train_preds_lr)
    train_report_rf = classification_report(y_train, y_train_preds_rf)

    # Generate classification report for testing data
    test_report_lr = classification_report(y_test, y_test_preds_lr)
    test_report_rf = classification_report(y_test, y_test_preds_rf)

    # Save LR classification reports as a single image
    combined_lr_report = f"Train Report (LR):\n\n{train_report_lr}\n\n" \
                        f"Test Report (LR):\n\n{test_report_lr}"
    # Create an image with white background
    image_width = 800
    image_height = 600
    background_color = (255, 255, 255)
    image = Image.new('RGB', (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)

    # Set font properties
    font_size = 20
    font_color = (0, 0, 0)
    font = ImageFont.truetype('arial.ttf', font_size)

    # Write the combined_lr_report text on the image
    text_position = (50, 50)
    draw.text(text_position, combined_lr_report, font=font, fill=font_color)

    # Save the image
    image.save('./images/results/generated_lr_report.png')


    # Save RF classification reports as a single image
    combined_rf_report = f"Train Report (RF):\n\n{train_report_rf}\n\n" \
                        f"Test Report (RF):\n\n{test_report_rf}"
    # Create an image with white background
    image_width = 800
    image_height = 600
    background_color = (255, 255, 255)
    image = Image.new('RGB', (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)

    # Set font properties
    font_size = 20
    font_color = (0, 0, 0)
    font = ImageFont.truetype('arial.ttf', font_size)

    # Write the combined_lr_report text on the image
    text_position = (50, 50)
    draw.text(text_position, combined_rf_report, font=font, fill=font_color)

    # Save the image
    image.save('./images/results/generated_rf_report.png')


    # Save the trained models
    joblib.dump(cv_rfc.best_estimator_, './models/random_forest_model.pkl')
    joblib.dump(lrc, './models/logistic_regression_model.pkl')

    # Calculate predicted probabilities for each class
    y_train_probs_lr = lrc.predict_proba(X_train)
    y_test_probs_lr = lrc.predict_proba(X_test)
    y_train_probs_rf = cv_rfc.best_estimator_.predict_proba(X_train)
    y_test_probs_rf = cv_rfc.best_estimator_.predict_proba(X_test)

    # Plot ROC curve for Logistic Regression model
    fig, axes = plt.subplots(figsize=(10, 6))
    roc_display_lr = RocCurveDisplay.from_estimator(
        lrc, X_test, y_test, ax=axes, name='LR')
    roc_display_lr.plot()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (LR)')
    plt.legend()
    plt.savefig('./images/results/roc_curve_lr.png')
    plt.close()

    # Plot ROC curve for Random Forest model
    fig, axes = plt.subplots(figsize=(10, 6))
    roc_display_rf = RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, X_test, y_test, ax=axes, name='RF')
    roc_display_rf.plot()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (RF)')
    plt.legend()
    plt.savefig('./images/results/roc_curve_rf.png')
    plt.close()

    return (
    cv_rfc.best_estimator_,
    lrc,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf
    )


if __name__ == "__main__":
    DF = import_data("./data/bank_data.csv")
    perform_eda(DF)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    DF = encoder_helper(DF, cat_columns)
    X_train2, X_test2, y_train2, y_test2 = perform_feature_engineering(DF)
    rfc2, lrc2, y_train_preds_lr2, y_train_preds_rf2, y_test_preds_lr2, y_test_preds_rf2 = train_models(
    X_train2,
    X_test2,
    y_train2,
    y_test2
    )
    feature_importance_plot(rfc2, DF, './images/results/')
    classification_report_image(
        y_train2,
        y_test2,
        y_train_preds_lr2,
        y_train_preds_rf2,
        y_test_preds_lr2,
        y_test_preds_rf2)
    