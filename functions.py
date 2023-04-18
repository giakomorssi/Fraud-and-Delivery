# Test the model
def prediction_100(model_name):
  from google.colab import drive
  drive.mount('/content/drive')
  from sklearn.metrics import mean_squared_error
  import pandas as pd
  from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
  from sklearn.model_selection import train_test_split
  from sklearn.decomposition import PCA
  from sklearn.metrics import confusion_matrix, recall_score
  import os
  import pickle
  import pandas as pd
  import numpy as np
  from category_encoders import LeaveOneOutEncoder

  within_threshold_mean = []
  threshold = 0.5
  mse = []

  df = pd.read_csv('/content/drive/MyDrive/University/Deloitte/df_lr.csv')
  X = df.drop(['Days for shipping (real)', 'Product Name'], axis = 1)
  y = df['Days for shipping (real)']

  # Load the specified model using pickle
  with open(f'/content/drive/MyDrive/University/Deloitte/models_lr/{model_name}.pkl', 'rb') as f:
      model = pickle.load(f)

  for i in range(1, 21):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    # initialize the encoder
    enc = LeaveOneOutEncoder(cols=['Customer City', 'Order City', 'Order State', 'Order Region'])

    # fit and transform the entire dataset
    X_train = enc.fit_transform(X_train, y_train)
    X_test = enc.transform(X_test)

    # fit and transform the entire dataset
    X_train = enc.fit_transform(X_train, y_train)
    X_test = enc.transform(X_test)

    # Select columns for one-hot encoding
    one_hot_cols = [0, 7, 9, 12, 15, 30]
    # Type, Department Name, Category Name, Market, Order Status, Customer Segment

    # Fit one-hot encoder to training data
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore")

    # Apply one-hot encoder to training and test data
    X_train_one_hot = one_hot_encoder.fit_transform(X_train.iloc[:, one_hot_cols])
    X_test_one_hot = one_hot_encoder.transform(X_test.iloc[:, one_hot_cols])

    # Remove original columns from training and test data
    X_train = X_train.drop(X_train.columns[one_hot_cols], axis=1)
    X_test = X_test.drop(X_test.columns[one_hot_cols], axis=1)

    le = LabelEncoder()
    # Shipping Mode
    custom_order = ['Same Day', 'First Class', 'Second Class', 'Standard Class']
    le.fit(custom_order)
    X_train['Shipping Mode'] = le.fit_transform(X_train['Shipping Mode'])
    X_test['Shipping Mode'] = le.transform(X_test['Shipping Mode'])

    # Delivery Status
    # Define the custom order
    custom_order = ['Shipping on time', 'Advance shipping', 'Late delivery', 'Shipping canceled']
    le.fit(custom_order)
    X_train['Delivery Status'] = le.fit_transform(X_train['Delivery Status'])
    X_test['Delivery Status'] = le.transform(X_test['Delivery Status'])

    # Concatenate one-hot encoded columns with remaining data
    X_train = pd.concat([pd.DataFrame(X_train_one_hot.toarray()), X_train.reset_index(drop=True)], axis=1)
    X_test = pd.concat([pd.DataFrame(X_test_one_hot.toarray()), X_test.reset_index(drop=True)], axis=1)

    scaler = StandardScaler()

    X_train.loc[:, X_train.columns[82:]] = scaler.fit_transform(X_train.loc[:, X_train.columns[82:]])
    X_test.loc[:, X_test.columns[82:]] = scaler.transform(X_test.loc[:, X_test.columns[82:]])

    # Split the dataset into features and target
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_train = np.ravel(y_train)

    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    y_pred = model.predict(X_test)
      
    mse.append(mean_squared_error(y_test, y_pred))

    # Calculate the percentage of predictions within the threshold value

    within_threshold_mean.append(sum(abs(y_pred - y_test) <= threshold) / len(y_pred))

  print(f'MSE Mean: {np.mean(mse)}')
  print(f'MSE Std: {np.std(mse)}')
  print(f'Within Threshold Mean: {np.mean(within_threshold_mean)}')
  print(f'Within Threshold Std: {np.std(within_threshold_mean)}')
