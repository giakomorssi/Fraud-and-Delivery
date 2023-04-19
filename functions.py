# Test the model
def prediction_100(model_name='rf', threshold= 0.5):
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
  mse = []

  df = pd.read_csv('/content/drive/MyDrive/University/Deloitte/df_lr.csv')
  X = df.drop(['Days for shipping (real)', 'Product Name'], axis = 1)
  y = df['Days for shipping (real)']

  # Load the specified model using pickle
  with open(f'/content/drive/MyDrive/University/Deloitte/models_lr/{model_name}.pkl', 'rb') as f:
      model = pickle.load(f)
      
  print('\nModel: \n', model, '\n')
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

  print(f'\nMSE Mean: {np.mean(mse)}')
  print(f'MSE Std: {np.std(mse)}')
  print(f'Within Threshold Mean: {np.mean(within_threshold_mean)}')
  print(f'Within Threshold Std: {np.std(within_threshold_mean)}')

def fraud_detection(model_name='rf', iteration=10):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
    import os
    import pickle
    import pandas as pd
    import numpy as np
    from category_encoders import LeaveOneOutEncoder
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder

    with open(f'/content/drive/MyDrive/University/Deloitte/model_fraud/{model_name}.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('/content/drive/MyDrive/University/Deloitte/df_fraud.csv')
    df.drop(['Order Status'], axis=1, inplace=True)

    X = df.drop(['Category'], axis=1)
    y = df['Category']

    # Standardize the data and split it into training and test sets
    recall_scores = []
    precision_scores = []
    fraud_recall = []
    suspected_recall = []
    regular_recall = [] 
    low = []
    avg_conf_matrix = np.zeros((3, 3))
    print('\nModel: \n', model, '\n')

    for i in range(1, (iteration + 1)):

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True)

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

      # Category
      y_train = le.fit_transform(y_train)
      y_test = le.transform(y_test)

      # initialize the encoder
      enc = LeaveOneOutEncoder(cols=['Order City'])

      # fit and transform the entire dataset
      X_train = enc.fit_transform(X_train, y_train)
      X_test = enc.transform(X_test)

      # Select columns for one-hot encoding
      one_hot_cols = [0, 7, 8, 11]

      # Fit one-hot encoder to training data
      one_hot_encoder = OneHotEncoder(handle_unknown="ignore")

      # Apply one-hot encoder to training and test data
      X_train_one_hot = one_hot_encoder.fit_transform(X_train.iloc[:, one_hot_cols])
      X_test_one_hot = one_hot_encoder.transform(X_test.iloc[:, one_hot_cols])

      # Remove original columns from training and test data
      X_train = X_train.drop(X_train.columns[one_hot_cols], axis=1)
      X_test = X_test.drop(X_test.columns[one_hot_cols], axis=1)

      # Concatenate one-hot encoded columns with remaining data
      X_train = pd.concat([pd.DataFrame(X_train_one_hot.toarray()), X_train.reset_index(drop=True)], axis=1)
      X_test = pd.concat([pd.DataFrame(X_test_one_hot.toarray()), X_test.reset_index(drop=True)], axis=1)

      X_train.columns = X_train.columns.astype(str)
      X_test.columns = X_test.columns.astype(str)

      s = StandardScaler()

      X_train[X_train.columns[23:]] = s.fit_transform(X_train[X_train.columns[23:]])
      X_test[X_test.columns[23:]] = s.transform(X_test[X_test.columns[23:]])

      X_train = pd.DataFrame(X_train)
      X_test = pd.DataFrame(X_test)
      y_train = pd.DataFrame(y_train)
      y_test = pd.DataFrame(y_test)
      y_train = np.ravel(y_train)

      X_train.columns = X_train.columns.astype(str)
      X_test.columns = X_test.columns.astype(str)

      # Initialize a PCA object
      pca = PCA()

      # Fit the PCA object to the data
      pca.fit(X_train.iloc[:, 23:])

      # Determine the number of components to keep
      variance_threshold = 0.95
      cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
      num_components_to_keep = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1

      # Transform the data using the chosen number of components
      pca = PCA(n_components=num_components_to_keep)
      pca_train = pca.fit_transform(X_train.iloc[:, 23:])
      pca_test = pca.transform(X_test.iloc[:, 23:])

      onehot_data_train = X_train.iloc[:, :23]
      onehot_data_test = X_test.iloc[:, :23]

      X_train = pd.concat([pd.DataFrame(pca_train), pd.DataFrame(onehot_data_train)], axis=1)
      X_test = pd.concat([pd.DataFrame(pca_test), pd.DataFrame(onehot_data_test)], axis=1)

      X_train = pd.DataFrame(X_train)
      X_test = pd.DataFrame(X_test)

      X_train.columns = X_train.columns.astype(str)
      X_test.columns = X_test.columns.astype(str)

      y_pred = model.predict(X_test)
      conf_matrix = confusion_matrix(y_test, y_pred)

      recall_scores.append(recall_score(y_test, y_pred, average=None))
      fraud_recall.append(recall_score(y_test, y_pred, average=None)[0])
      regular_recall.append(recall_score(y_test, y_pred, average=None)[1])
      suspected_recall.append(recall_score(y_test, y_pred, average=None)[2])
      precision_scores.append(precision_score(y_test, y_pred, average=None))

      conf_matrix = confusion_matrix(y_test, y_pred)
      avg_conf_matrix += conf_matrix

      if i % 10 == 0:
          print(f'Iteration: {i}')

      if recall_score(y_test, y_pred, average=None)[0] < 0.7:
        low.append(round(recall_score(y_test, y_pred, average=None)[0], 4))

    print(f'\n Fraud Recall: {round(np.average(fraud_recall), 4)}, std: {round(np.std(fraud_recall), 4)}, Under 0.7: {len(low)}, {low}\n Suspected Recall: {round(np.average(suspected_recall), 4)}, std: {round(np.std(suspected_recall), 4)}\n Regular Recall: {round(np.average(regular_recall), 4)}, std: {round(np.std(regular_recall), 4)}\n Total: {round(np.average(recall_scores), 4)}, std: {round(np.std(recall_scores), 4)}')

    np.set_printoptions(precision=4)
    avg_conf_matrix /= iteration
    print('Precisions', precision_score(y_test, y_pred, average=None))
    print("\n Average Confusion Matrix:")
    print(avg_conf_matrix)

def visual_prediction(model_name = 'rf', threshold = 0.5):
  import seaborn as sns
  import matplotlib.pyplot as plt
  import statsmodels.api as sm
  from statsmodels.graphics.gofplots import qqplot
  from google.colab import drive
  drive.mount('/content/drive')
  from sklearn.metrics import mean_squared_error
  from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
  from sklearn.model_selection import train_test_split
  from sklearn.decomposition import PCA
  from sklearn.metrics import confusion_matrix, recall_score, mean_squared_error
  import os
  import pickle
  import pandas as pd
  import numpy as np
  from category_encoders import LeaveOneOutEncoder

  with open(f'/content/drive/MyDrive/University/Deloitte/models_lr/{model_name}.pkl', 'rb') as f:
    model = pickle.load(f)
    
   print(f'\nModel: \n{model} \n')

  df = pd.read_csv('/content/drive/MyDrive/University/Deloitte/df_lr.csv')
  X = df.drop(['Days for shipping (real)', 'Product Name'], axis = 1)
  y = df['Days for shipping (real)']
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

  print('Model: ', model_name, '\n')
      
  print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False), '\n')

  residuals = y_test - y_pred

  # Distribution plot
  fig, ax = plt.subplots(figsize=(10, 6))
  sns.set_palette('colorblind')
  sns.set_style('darkgrid')
  sns.kdeplot(y_test, label='Actual', fill=True)
  sns.kdeplot(y_pred, label='Predicted', fill=True)
  plt.xlabel('Delivery time')
  plt.ylabel('Density')
  plt.title('Distribution Plot')
  plt.legend()
  plt.show()
  print('\n')

  # Plot a histogram of the residuals
  sns.set_style('darkgrid')
  fig, ax = plt.subplots(figsize=(10, 6))
  sns.histplot(residuals, bins=100, color='green', kde=True, ax=ax)
  ax.set_xlabel('Residuals')
  ax.set_ylabel('Frequency')
  ax.set_title('Histogram of Residuals')
  ax.set_xlim((-3, 3))
  sns.despine()
  plt.show()
  print('\n')

  # Plot the percentage of predictions within threshold days
  y_pred = model.predict(X_test)
  within_threshold = (np.abs(y_pred - y_test) < threshold).mean()
  outside_threshold = 1 - within_threshold
  fig, ax = plt.subplots()
  labels = ['Within threshold', 'Outside threshold']
  sizes = [within_threshold, outside_threshold]
  colors = ['#1f77b4', '#ff7f0e']
  plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
  plt.title('Percentage of Predictions Within Threshold', fontsize=13)
  plt.show()
  print('\n')

  # QQ plot
  residuals = y_test - y_pred
  qqplot(residuals, line='s')
  plt.title('QQ Plot of Residuals')
  plt.xlabel('Theoretical Quantiles')
  plt.ylabel('Sample Quantiles')
  plt.show()
