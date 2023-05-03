def prediction_pkl_st(model, df, threshold = 0.1):
  import streamlit as st
  from sklearn.metrics import mean_squared_error
  import pandas as pd
  from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
  from sklearn.model_selection import train_test_split
  from sklearn.decomposition import PCA
  from sklearn.metrics import confusion_matrix, recall_score, mean_absolute_error, r2_score
  import os
  import pickle
  import pandas as pd
  import numpy as np
  from category_encoders import LeaveOneOutEncoder
  from keras.models import load_model
  from matplotlib import pyplot as plt
  import seaborn as sns
  from statsmodels.graphics.gofplots import qqplot
  import statsmodels.api as sm
  
  within_threshold_mean = []
  mse_v = []
  mae_v = []
  r2_score_v = []

  with st.spinner('Wait for it...'):
    st.subheader('\nModel:\n')
    st.write(model)
   
  st.write(f'The Within Threshold is: {threshold} that corresponds to {np.round(24*threshold, 2)} hours.')

  with st.spinner('Running prediction...'):
    progress_bar = st.progress(0)

    for i in range(1, 11):
      df = df.sample(frac=1)

      X = df.drop(['Days for shipping (real)'], axis = 1)
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
      one_hot_cols = [0, 3, 5, 7, 10]
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

      # Concatenate one-hot encoded columns with remaining data
      X_train = pd.concat([pd.DataFrame(X_train_one_hot.toarray()), X_train.reset_index(drop=True)], axis=1)
      X_test = pd.concat([pd.DataFrame(X_test_one_hot.toarray()), X_test.reset_index(drop=True)], axis=1)

      scaler = StandardScaler()

      X_train[X_train.columns[73:]] = scaler.fit_transform(X_train[X_train.columns[73:]])
      X_test[X_test.columns[73:]] = scaler.transform(X_test[X_test.columns[73:]])

      # Split the dataset into features and target
      X_train = pd.DataFrame(X_train)
      X_test = pd.DataFrame(X_test)
      y_train = pd.DataFrame(y_train)
      y_train = np.ravel(y_train)

      X_train.columns = X_train.columns.astype(str)
      X_test.columns = X_test.columns.astype(str)

      y_pred = model.predict(X_test)

      progress_bar.progress((i + 1) / 11)

      # Calculate the percentage of predictions within the threshold value
      
      mse = mean_squared_error(y_test, y_pred)
      mae = mean_absolute_error(y_test, y_pred)
      r2_score_v.append(r2_score(y_test, y_pred))
      
      mae_v.append(mae)
      mse_v.append(mse)
      
      within_threshold_mean.append(sum(abs(y_pred.ravel() - y_test.ravel()) <= threshold) / len(y_pred))

  st.subheader('Model Performance')
  table_header = ['Metric', 'Mean']
  table_data = [
      ['rMSE', f'{np.mean(np.sqrt(mse_v)):.4f}'],
      ['MAE', f'{np.mean(mae_v):.4f}'],
      ['MSE', f'{np.mean(mse_v):.4f}'],
      ['R2', f'{np.mean(r2_score_v):.4f}'],
      ['Within Threshold', f'{np.mean(within_threshold_mean):.4f}']
  ]

  st.table(pd.DataFrame(table_data, columns=table_header))
    
  y_test = np.ravel(y_test)
  y_pred = np.ravel(y_pred)
  
  residuals = y_test - y_pred

  # Initialize a dictionary to store the squared errors and average predictions for each class
  mse_dict = {label: {'errors': [], 'preds': []} for label in np.unique(y_test)}

  # Calculate the squared error and average prediction for each instance and add them to the corresponding class lists
  for true_label, pred_label, error in zip(y_test, y_pred, (y_test - y_pred) ** 2):
      mse_dict[true_label]['errors'].append(error)
      mse_dict[true_label]['preds'].append(pred_label)

  st.subheader('MSE and Average Prediction for Each Day')
  table_header = ['Class', 'Instances', 'RMSE', 'Average Predicted Value']
  table_data = []
  # Calculate the mean squared error and average prediction for each class and add the results to the table_data list
  for label in mse_dict:
      instances = len(mse_dict[label]['errors'])
      if instances > 0:
          mse = np.mean(mse_dict[label]['errors'])
          rmse = np.sqrt(mse)
          avg_pred = np.mean(mse_dict[label]['preds'])
          table_data.append([f'{label} Days', instances, f'{rmse:.4f}', f'{avg_pred:.4f}'])
      else:
          table_data.append([f'{label} Day', 'no instances', 'N/A', 'N/A'])

  st.table(pd.DataFrame(table_data, columns=table_header))


  st.subheader('Distribution Plot of Prediction vs. Actual')
  # Distribution plot
  fig, ax = plt.subplots(figsize=(10, 6))
  sns.set_palette('colorblind')
  sns.set_style('darkgrid')
  sns.kdeplot(y_test.ravel(), label='Actual', fill=True)
  sns.kdeplot(y_pred.ravel(), label='Predicted', fill=True)
  plt.xlabel('Delivery time')
  plt.ylabel('Density')
  plt.title('Distribution Plot')
  plt.legend()
  st.pyplot(fig)
  st.write('\n')

  st.subheader('Histogram of the Residuals')
  # Plot a histogram of the residuals
  sns.set_style('darkgrid')
  fig, ax = plt.subplots(figsize=(10, 6))
  sns.histplot(residuals, bins=100, color='green', kde=True, ax=ax)
  ax.set_xlabel('Residuals')
  ax.set_ylabel('Frequency')
  ax.set_title('Histogram of Residuals')
  ax.set_xlim((-1.5, 1.5))
  sns.despine()
  st.pyplot(fig)
  st.write('\n')

  st.subheader('Percentage of Predictions within Half Day')
  # Plot the percentage of predictions within threshold days
  y_pred = model.predict(X_test)
  within_threshold = (np.abs(y_pred.ravel() - y_test.ravel()) < threshold).mean()
  outside_threshold = 1 - within_threshold
  fig, ax = plt.subplots()
  labels = ['Within threshold', 'Outside threshold']
  sizes = [within_threshold, outside_threshold]
  colors = ['#1f77b4', '#ff7f0e']
  plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
  plt.title('Percentage of Predictions Within Threshold', fontsize=13)
  st.pyplot(fig)
  st.write('\n')

  st.subheader('QQ Plot')
  # QQ plot
  fig, ax = plt.subplots()
  sm.qqplot(residuals, line='s', ax=ax)
  ax.set_title('QQ Plot')
  plt.xlabel('Theoretical Quantiles')
  plt.ylabel('Sample Quantiles')

  # Show the plot in Streamlit
  st.pyplot(fig)

def fraud_detection_st(model, df):
      import streamlit as st
      import seaborn as sns
      import matplotlib.pyplot as plt
      from sklearn.preprocessing import StandardScaler
      from sklearn.model_selection import train_test_split
      from sklearn.decomposition import PCA
      from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
      import os
      import matplotlib.pyplot as plt
      import pickle
      import pandas as pd
      import numpy as np
      from category_encoders import LeaveOneOutEncoder
      from sklearn.preprocessing import OneHotEncoder, LabelEncoder
      import graphviz
      from sklearn.tree import export_graphviz
      
      # Standardize the data and split it into training and test sets
      recall_scores = []
      precision_scores = []
      fraud_recall = []
      fraud_precision = []
      suspected_recall = []
      suspected_precision = []
      regular_recall = [] 
      regular_precision = []
      avg_conf_matrix = np.zeros((3, 3))
      with st.spinner('Wait for it...'):
        st.subheader('\nModel:\n')
        st.write(model)

      with st.spinner('Running prediction...'):
        progress_bar = st.progress(0)
        for i in range(1, (11)):
          df = df.sample(frac=1)

          X = df.drop(['Category'], axis=1)
          y = df['Category']
          
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
          X_train.iloc[:, 23:] = s.fit_transform(X_train.iloc[:, 23:])
          X_test.iloc[:, 23:] = s.transform(X_test.iloc[:, 23:])

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
          
          X_train.columns = range(len(X_train.columns))
          X_test.columns = range(len(X_test.columns))

          y_pred = model.predict(X_test)

          progress_bar.progress((i + 1) / 11)
          conf_matrix = confusion_matrix(y_test, y_pred)

          recall_scores.append(recall_score(y_test, y_pred, average='macro'))
          fraud_recall.append(recall_score(y_test, y_pred, average=None)[0])
          regular_recall.append(recall_score(y_test, y_pred, average=None)[1])
          suspected_recall.append(recall_score(y_test, y_pred, average=None)[2])
          precision_scores.append(precision_score(y_test, y_pred, average='macro'))
          fraud_precision.append(precision_score(y_test, y_pred, average=None)[0])
          regular_precision.append(precision_score(y_test, y_pred, average=None)[1])
          suspected_precision.append(precision_score(y_test, y_pred, average=None)[2])
          
          conf_matrix = confusion_matrix(y_test, y_pred)
          avg_conf_matrix += conf_matrix

      st.subheader('Recall Performance')

      table_header = ['Recall Type', 'Average', 'Std']
      table_data = [    ['Fraud', f'{np.average(fraud_recall):.4f}', f'{np.std(fraud_recall):.4f}'],
          ['Suspected', f'{np.average(suspected_recall):.4f}', f'{np.std(suspected_recall):.4f}'],
          ['Regular', f'{np.average(regular_recall):.4f}', f'{np.std(regular_recall):.4f}'],
          ['Total', f'{np.average(recall_scores):.4f}', f'{np.std(recall_scores):.4f}']
      ]
      st.table(pd.DataFrame(table_data, columns=table_header))

      st.subheader('Precision Performance')

      table_header = ['Precision Type', 'Average', 'Std']
      table_data = [    ['Fraud', f'{np.average(fraud_precision):.4f}', f'{np.std(fraud_precision):.4f}'],
          ['Suspected', f'{np.average(suspected_precision):.4f}', f'{np.std(suspected_precision):.4f}'],
          ['Regular', f'{np.average(regular_precision):.4f}', f'{np.std(regular_precision):.4f}'],
          ['Total', f'{np.average(precision_scores):.4f}', f'{np.std(precision_scores):.4f}']
      ]
      st.table(pd.DataFrame(table_data, columns=table_header))


      st.subheader('Confusion Matrix')

      # plot the confusion matrix using seaborn heatmap
      pd.options.display.float_format = '{:.1f}'.format
      fig, ax = plt.subplots(figsize=(8, 6))
      avg_conf_matrix /= 10
      np.set_printoptions(precision=1, suppress=True)
      avg_conf_matrix = np.round(avg_conf_matrix).astype(int)
      sns.heatmap(avg_conf_matrix, annot=True, cmap='Blues', fmt='.1f', cbar=False)
      plt.xlabel('Predicted')
      plt.ylabel('Actual')
      plt.title('Confusion Matrix')
      st.pyplot(fig)

def delay_detection_st(model, df):
      import streamlit as st
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
      import matplotlib.pyplot as plt
      import seaborn as sns

      # Standardize the data and split it into training and test sets
      recall_scores = []
      precision_scores = []
      accuracy_scores = []
      f1_scores = []
      avg_conf_matrix = np.zeros((2, 2))

      st.subheader('\nModel:\n')
      st.write(model)

      with st.spinner('Running prediction...'):
          progress_bar = st.progress(0)
          for i in range(1, (10 + 1)):
            df = df.sample(frac=1)
      
            X = df.drop('Delay', axis = 1)
            y = df['Delay']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True)

            # initialize the encoder
            enc = LeaveOneOutEncoder(cols=['Customer City', 'Order City'])

            # fit and transform the entire dataset
            X_train = enc.fit_transform(X_train, y_train)
            X_test = enc.transform(X_test)

            # Select columns for one-hot encoding
            one_hot_cols = [0, 3, 5, 6, 9]
            one_hot_encoder = OneHotEncoder(handle_unknown="ignore")
            X_train_one_hot = one_hot_encoder.fit_transform(X_train.iloc[:, one_hot_cols])
            X_test_one_hot = one_hot_encoder.transform(X_test.iloc[:, one_hot_cols])
            X_train = X_train.drop(X_train.columns[one_hot_cols], axis=1)
            X_test = X_test.drop(X_test.columns[one_hot_cols], axis=1)
            X_train = pd.concat([pd.DataFrame(X_train_one_hot.toarray()), X_train.reset_index(drop=True)], axis=1)
            X_test = pd.concat([pd.DataFrame(X_test_one_hot.toarray()), X_test.reset_index(drop=True)], axis=1)

            le = LabelEncoder()

            # Shipping Mode
            custom_order = ['Same Day', 'First Class', 'Second Class', 'Standard Class']
            le.fit(custom_order)
            X_train['Shipping Mode'] = le.fit_transform(X_train['Shipping Mode'])
            X_test['Shipping Mode'] = le.transform(X_test['Shipping Mode'])

            X_train.columns = X_train.columns.astype(str)
            X_test.columns = X_test.columns.astype(str)

            scaler = StandardScaler()

            X_train[X_train.columns[73:]] = scaler.fit_transform(X_train[X_train.columns[73:]])
            X_test[X_test.columns[73:]] = scaler.transform(X_test[X_test.columns[73:]])

            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
            y_train = pd.DataFrame(y_train)
            y_train = np.ravel(y_train)

            X_train.columns = X_train.columns.astype(str)
            X_test.columns = X_test.columns.astype(str)

            progress_bar.progress((i + 1) / 11)
            y_pred = model.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)

            recall_scores.append(recall_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))

            conf_matrix = confusion_matrix(y_test, y_pred)
            avg_conf_matrix += conf_matrix

      st.subheader('Performance')

      table_header = ['Score Type', 'Average', 'Std']
      table_data = [
          ['Recall', f'{round(np.average(recall_scores), 4)}', f'{round(np.std(recall_scores), 6)}'],
          ['Precision', f'{round(np.average(precision_scores), 4)}', f'{round(np.std(precision_scores), 6)}'],
          ['F1', f'{round(np.average(f1_scores), 4)}', f'{round(np.std(f1_scores), 6)}'],
          ['Accuracy', f'{round(np.average(accuracy_scores), 4)}', f'{round(np.std(accuracy_scores), 6)}']
      ]

      st.table(pd.DataFrame(table_data, columns=table_header))

      st.subheader('Confusion Matrix')

      pd.options.display.float_format = '{:.1f}'.format
      fig, ax = plt.subplots(figsize=(8, 6))
      avg_conf_matrix /= 10
      np.set_printoptions(precision=1, suppress=True)
      avg_conf_matrix = np.round(avg_conf_matrix).astype(int)
      sns.heatmap(avg_conf_matrix, annot=True, cmap='Blues', fmt='.1f', cbar=False)
      plt.xlabel('Predicted')
      plt.ylabel('Actual')
      plt.title('Confusion Matrix')
      st.pyplot(fig)
