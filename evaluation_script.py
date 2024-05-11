import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, median_absolute_error, r2_score
import numpy as np

def preprocessing_data(filename):
    """
    Preprocesses the raw data from the CSV file.
    
    Parameters:
    filename (str): Path to the CSV file containing the raw data.
    
    Returns:
    x (DataFrame): Processed feature matrix.
    y (Series): Target variable.
    """
    raw_data = pd.read_csv(filename)
    raw_data = raw_data.drop(columns=['dataset_transaction', 'dataset_user'])
    raw_data['mcc_group'] = raw_data['mcc_group'].fillna(0).astype(int)
    raw_data['transaction_date'] = pd.to_datetime(raw_data['transaction_date'])
    raw_data['transaction_day'] = raw_data['transaction_date'].dt.day
    raw_data = raw_data.drop(columns=['transaction_date'])
    
    with open('./encoder/label_encoder.pkl', 'rb') as f:
        loaded_label_encoder = pickle.load(f)
    loaded_label_encoder.fit(raw_data['user_id'])
    raw_data['user_id'] = loaded_label_encoder.transform(raw_data['user_id'])

    one_hot_encoded_type = pd.get_dummies(raw_data['transaction_type'], prefix='transaction')
    one_hot_encoded_mcc_group = pd.get_dummies(raw_data['mcc_group'], prefix='mcc_group')
    one_hot_encoded_type = one_hot_encoded_type.astype(int)
    one_hot_encoded_mcc_group = one_hot_encoded_mcc_group.astype(int)
    raw_data = pd.concat([raw_data, one_hot_encoded_type, one_hot_encoded_mcc_group], axis=1)
    raw_data = raw_data.drop(columns=['transaction_type', 'mcc_group'])

    y = raw_data['amount_currency']
    x = raw_data.drop(columns=['amount_currency'])
    return x, y 

def predict_model(x, y):
    """
    Predicts the target variable using the trained model.
    
    Parameters:
    x (DataFrame): Feature matrix.
    y (Series): Target variable.
    
    Returns:
    y_pred (array): Predicted target variable.
    """
    with open('./model/final_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    y_pred = loaded_model.predict(x)
    return y_pred

def evaluate_model(y_pred, y):
    """
    Evaluates the performance of the model using various metrics.
    
    Parameters:
    y_pred (array): Predicted target variable.
    y (Series): Actual target variable.
    """
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    explained_variance = explained_variance_score(y, y_pred)
    medae = median_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("**Evaluation Metrics:**")
    print(f"· Mean Squared Error (MSE): {mse:.2f}")
    print(f"· Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"· Mean Absolute Error (MAE): {mae:.2f}")
    print(f"· Explained Variance Score: {explained_variance:.2f}")
    print(f"· Median Absolute Error (MedAE): {medae:.2f}")
    print(f"· R-squared (R2) Score: {r2:.2f}")

def main():
    """
    Main function to preprocess data, predict, and evaluate the model.
    """
    filename = input("Enter the path to the CSV file: ")
    x , y = preprocessing_data(filename)
    y_pred = predict_model(x, y)
    evaluate_model(y_pred, y)
    
if __name__ == "__main__":
    main()
