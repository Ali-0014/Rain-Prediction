#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import opendatasets as od
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Downloading the data

# In[2]:


dataset_url = 'https://www.kaggle.com/jsphyg/weather-dataset-rattle-package'


# In[3]:


od.download(dataset_url)


# In[4]:


import os


# In[5]:


data_dir = './weather-dataset-rattle-package'
train_csv = data_dir + './weatherAUS.csv'


# In[6]:


raw_df = pd.read_csv(train_csv)


# In[7]:


raw_df


# In[8]:


raw_df.info()


# In[9]:


raw_df.dropna(subset=['RainToday','RainTomorrow'],inplace = True)


# # Exploratory Data Analysis

# In[10]:


px.histogram(raw_df, x = 'Location', title='Location vs. Rainy Days', color='RainToday')


# In[11]:


px.histogram(raw_df, x='Temp3pm', title='Temperature at 3 pm vs. Rain Tomorrow', color='RainTomorrow')


# In[12]:


px.histogram(raw_df, x='RainTomorrow', color='RainToday', title='Rain Tomorrow vs. Rain Today')


# In[13]:


px.scatter(raw_df.sample(2000), title='Min Temp. vs Max Temp.',x='MinTemp',y='MaxTemp', color='RainToday')


# In[14]:


px.scatter(raw_df.sample(2000), title='Temp (3 pm) vs. Humidity (3 pm)',x='Temp3pm',y='Humidity3pm',color='RainTomorrow')


# In[15]:


top_locations = raw_df['Location'].value_counts().head(10).index
filtered_data = raw_df[raw_df['Location'].isin(top_locations)].sample(5000)
plt.figure(figsize=(12,8))
sns.boxplot(data=filtered_data, x='Location', y='MaxTemp')
plt.title('Max Temperature Distribution by Top 10 Locations')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # Training, Validation and Test Sets

# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)


# In[18]:


print(train_df.shape)
print(val_df.shape)
print(test_df.shape)


# In[19]:


plt.title('No. of Rows per Year')
sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year)


# In[20]:


# Create training, validation and test sets
year = pd.to_datetime(raw_df.Date).dt.year
train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]


# In[21]:


print(train_df.shape)
print(val_df.shape)
print(test_df.shape)


# In[22]:


train_df


# # Identifying Input and Target Columns
# 

# In[23]:


# Create inputs and targets
input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'


# In[24]:


print(input_cols)
print(target_col)


# In[25]:


train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()


# In[26]:


val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()


# In[27]:


test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()


# In[28]:


train_inputs


# In[29]:


train_targets


# In[30]:


# Identify numeric and categorical columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


# In[31]:


numeric_cols


# In[32]:


categorical_cols


# In[33]:


train_inputs[numeric_cols].describe()


# In[34]:


train_inputs[categorical_cols].nunique()


# # Imputing Missing Numeric Data

# In[35]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')


# In[36]:


raw_df[numeric_cols].isna().sum()


# In[37]:


train_inputs[numeric_cols].isna().sum()


# In[38]:


# Impute missing numerical values
imputer.fit(raw_df[numeric_cols])


# In[39]:


list(imputer.statistics_)


# In[40]:


train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])


# In[41]:


train_inputs[numeric_cols].isna().sum()


# In[42]:


raw_df[numeric_cols].describe()


# # Scaling Numeric Features

# In[43]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(raw_df[numeric_cols])


# In[44]:


list(scaler.data_min_)


# In[45]:


list(scaler.data_max_)


# In[46]:


train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


# # Encoding Categorical Data

# In[47]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output = False, handle_unknown= 'ignore')
encoder.fit(raw_df[categorical_cols])


# In[48]:


encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


# # Training a Logistic Regression Model

# In[49]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix


# In[50]:


X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


# In[51]:


model = LogisticRegression(solver='liblinear')
model.fit(X_train, train_targets)


# In[52]:


train_preds = model.predict(X_train)
train_probs = model.predict_proba(X_train)
accuracy_score(train_targets, train_preds)


# # Results and Evaluation

# In[53]:


def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average='weighted')  
    recall = recall_score(targets, preds, average='weighted')        
    f1 = f1_score(targets, preds, average='weighted')               
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True, fmt=".2f", cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title(f'{name} Confusion Matrix')
    
    return preds


# In[54]:


train_preds = predict_and_plot(X_train, train_targets, 'Training' )


# In[55]:


val_preds = predict_and_plot(X_val, val_targets, 'Validation' )


# In[56]:


test_preds = predict_and_plot(X_test, test_targets, 'Test')


# In[57]:


def plot_scores(accuracy, precision, recall, f1, name=''):
    scores = [accuracy, precision, recall, f1]
    score_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    plt.figure(figsize=(8, 6))
    bars = plt.bar(score_names, scores, color=['blue', 'orange', 'green', 'red'])
    plt.ylim(0, 1)
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                 f"{score:.2f}", ha='center', va='bottom', fontsize=12)
    
    plt.title(f'{name} Set Performance Metrics', fontsize=16)
    plt.ylabel('Scores', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)
    plt.show()

val_accuracy = accuracy_score(val_targets, val_preds)
val_precision = precision_score(val_targets, val_preds, average='weighted')
val_recall = recall_score(val_targets, val_preds, average='weighted')
val_f1 = f1_score(val_targets, val_preds, average='weighted')


plot_scores(val_accuracy, val_precision, val_recall, val_f1, name='Validation')


# In[58]:


test_accuracy = accuracy_score(test_targets, test_preds)
test_precision = precision_score(test_targets, test_preds, average='weighted')
test_recall = recall_score(test_targets, test_preds, average='weighted')
test_f1 = f1_score(test_targets, test_preds, average='weighted')

plot_scores(test_accuracy, test_precision, test_recall, test_f1, name='Test')


# # Prediction on Single Input

# In[59]:


def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob

new_input = {'Date': '2021-06-19',
             'Location': 'Launceston',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}

predict_input(new_input)


# # Model comparsion

# In[68]:


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

scaler_results = {}

def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds, average='weighted')
    recall = recall_score(y_val, val_preds, average='weighted')
    f1 = f1_score(y_val, val_preds, average='weighted')
    return accuracy, precision, recall, f1

models = {
    'LogisticRegression': LogisticRegression(solver='liblinear'),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=40),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=40),
    'SVM': SVC(kernel='linear', max_iter=5000),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, random_state=40),  
}

for scaler_name, scaler in [('StandardScaler', StandardScaler()), ('MinMaxScaler', MinMaxScaler())]:
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    
    results = {}
    
    
    for name, model in models.items():
        accuracy, precision, recall, f1 = evaluate_model(model, X_train_scaled, train_targets, X_val_scaled, val_targets)
        results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    
    scaler_results[scaler_name] = pd.DataFrame(results).T

plt.figure(figsize=(20, 10))

for i, (scaler_name, results_df) in enumerate(scaler_results.items(), 1):
    plt.subplot(1, 2, i)
    results_df.plot(kind='bar', figsize=(20, 10), rot=0, ax=plt.gca())
    plt.title(f'Model Comparison ({scaler_name}) and Mean Imputer')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.grid(axis='y')
    plt.tight_layout()

plt.show()

for scaler_name, results_df in scaler_results.items():
    print(f"Results for {scaler_name}:\n")
    print(results_df, "\n")


# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

results = {
    'Mean': {
        'StandardScaler': {
            'LogisticRegression': {'Accuracy': 0.854058, 'Precision': 0.844006, 'Recall': 0.854058, 'F1 Score': 0.840511},
            'KNN': {'Accuracy': 0.818772, 'Precision': 0.797329, 'Recall': 0.818772, 'F1 Score': 0.795123},
            'Random Forest': {'Accuracy': 0.856223, 'Precision': 0.848053, 'Recall': 0.856223, 'F1 Score': 0.840177},
            'Naive Bayes': {'Accuracy': 0.668325, 'Precision': 0.775564, 'Recall': 0.668325, 'F1 Score': 0.697891},
            'Decision Tree': {'Accuracy': 0.793493, 'Precision': 0.792542, 'Recall': 0.793493, 'F1 Score': 0.793011},
            'SVM': {'Accuracy': 0.592135, 'Precision': 0.699412, 'Recall': 0.592135, 'F1 Score': 0.627879},
            'Neural Network': {'Accuracy': 0.865645, 'Precision': 0.857655, 'Recall': 0.865645, 'F1 Score': 0.855612}
        },
        'MinMaxScaler': {
            'LogisticRegression': {'Accuracy': 0.854058, 'Precision': 0.844054, 'Recall': 0.854058, 'F1 Score': 0.840328},
            'KNN': {'Accuracy': 0.821874, 'Precision': 0.802695, 'Recall': 0.821874, 'F1 Score': 0.803376},
            'Random Forest': {'Accuracy': 0.855814, 'Precision': 0.847592, 'Recall': 0.855814, 'F1 Score': 0.839595},
            'Naive Bayes': {'Accuracy': 0.668325, 'Precision': 0.775564, 'Recall': 0.668325, 'F1 Score': 0.697891},
            'Decision Tree': {'Accuracy': 0.793610, 'Precision': 0.792912, 'Recall': 0.793610, 'F1 Score': 0.793258},
            'SVM': {'Accuracy': 0.692434, 'Precision': 0.748625, 'Recall': 0.692434, 'F1 Score': 0.712985},
            'Neural Network': {'Accuracy': 0.862134, 'Precision': 0.853863, 'Recall': 0.862134, 'F1 Score': 0.850067}
        }
    },
    'Median': {
        'StandardScaler': {
            'LogisticRegression': {'Accuracy': 0.853590, 'Precision': 0.843456, 'Recall': 0.853590, 'F1 Score': 0.839842},
            'KNN': {'Accuracy': 0.818948, 'Precision': 0.797554, 'Recall': 0.818948, 'F1 Score': 0.795149},
            'Random Forest': {'Accuracy': 0.856165, 'Precision': 0.847750, 'Recall': 0.856165, 'F1 Score': 0.840460},
            'Naive Bayes': {'Accuracy': 0.669027, 'Precision': 0.776143, 'Recall': 0.669027, 'F1 Score': 0.698525},
            'Decision Tree': {'Accuracy': 0.795073, 'Precision': 0.796811, 'Recall': 0.795073, 'F1 Score': 0.795922},
            'SVM': {'Accuracy': 0.549593, 'Precision': 0.694073, 'Recall': 0.549593, 'F1 Score': 0.591494},
            'Neural Network': {'Accuracy': 0.864299, 'Precision': 0.856419, 'Recall': 0.864299, 'F1 Score': 0.852740}
        },
        'MinMaxScaler': {
            'LogisticRegression': {'Accuracy': 0.853473, 'Precision': 0.843379, 'Recall': 0.853473, 'F1 Score': 0.839449},
            'KNN': {'Accuracy': 0.823044, 'Precision': 0.804186, 'Recall': 0.823044, 'F1 Score': 0.804568},
            'Random Forest': {'Accuracy': 0.855989, 'Precision': 0.847492, 'Recall': 0.855989, 'F1 Score': 0.840307},
            'Naive Bayes': {'Accuracy': 0.669027, 'Precision': 0.776143, 'Recall': 0.669027, 'F1 Score': 0.698525},
            'Decision Tree': {'Accuracy': 0.794839, 'Precision': 0.796750, 'Recall': 0.794839, 'F1 Score': 0.795770},
            'SVM': {'Accuracy': 0.595588, 'Precision': 0.727057, 'Recall': 0.595588, 'F1 Score': 0.632908},
            'Neural Network': {'Accuracy': 0.862777, 'Precision': 0.854685, 'Recall': 0.862777, 'F1 Score': 0.850707}
        }
    },
    'Min': {
        'StandardScaler': {
            'LogisticRegression': {'Accuracy': 0.848441, 'Precision': 0.837251, 'Recall': 0.848441, 'F1 Score': 0.837762},
            'KNN': {'Accuracy': 0.809234, 'Precision': 0.783197, 'Recall': 0.809234, 'F1 Score': 0.782511},
            'Random Forest': {'Accuracy': 0.855814, 'Precision': 0.847498, 'Recall': 0.855814, 'F1 Score': 0.839735},
            'Naive Bayes': {'Accuracy': 0.638715, 'Precision': 0.765293, 'Recall': 0.638715, 'F1 Score': 0.671962},
            'Decision Tree': {'Accuracy': 0.791737, 'Precision': 0.792617, 'Recall': 0.791737, 'F1 Score': 0.792172},
            'SVM': {'Accuracy': 0.579086, 'Precision': 0.721662, 'Recall': 0.579086, 'F1 Score': 0.618232},
            'Neural Network': {'Accuracy': 0.855931, 'Precision': 0.846191, 'Recall': 0.855931, 'F1 Score': 0.846283}
        },
        'MinMaxScaler': {
            'LogisticRegression': {'Accuracy': 0.848616, 'Precision': 0.837441, 'Recall': 0.848616, 'F1 Score': 0.837889},
            'KNN': {'Accuracy': 0.816900, 'Precision': 0.796125, 'Recall': 0.816900, 'F1 Score': 0.797661},
            'Random Forest': {'Accuracy': 0.856048, 'Precision': 0.847754, 'Recall': 0.856048, 'F1 Score': 0.840080},
            'Naive Bayes': {'Accuracy': 0.638715, 'Precision': 0.765293, 'Recall': 0.638715, 'F1 Score': 0.671962},
            'Decision Tree': {'Accuracy': 0.791679, 'Precision': 0.792624, 'Recall': 0.791679, 'F1 Score': 0.792146},
            'SVM': {'Accuracy': 0.605360, 'Precision': 0.725763, 'Recall': 0.605360, 'F1 Score': 0.641218},
            'Neural Network': {'Accuracy': 0.856516, 'Precision': 0.846929, 'Recall': 0.856516, 'F1 Score': 0.847155}
        }
    },
    'Max': {
        'StandardScaler': {
            'LogisticRegression': {'Accuracy': 0.814735, 'Precision': 0.791840, 'Recall': 0.814735, 'F1 Score': 0.779715},
            'KNN': {'Accuracy': 0.797238, 'Precision': 0.766901, 'Recall': 0.797238, 'F1 Score': 0.770907},
            'Random Forest': {'Accuracy': 0.856282, 'Precision': 0.847802, 'Recall': 0.856282, 'F1 Score': 0.840756},
            'Naive Bayes': {'Accuracy': 0.626251, 'Precision': 0.750321, 'Recall': 0.626251, 'F1 Score': 0.660496},
            'Decision Tree': {'Accuracy': 0.795482, 'Precision': 0.794083, 'Recall': 0.795482, 'F1 Score': 0.794769},
            'SVM': {'Accuracy': 0.502604, 'Precision': 0.686760, 'Recall': 0.502604, 'F1 Score': 0.547457},
            'Neural Network': {'Accuracy': 0.858447, 'Precision': 0.849118, 'Recall': 0.858447, 'F1 Score': 0.848845}
        },
        'MinMaxScaler': {
            'LogisticRegression': {'Accuracy': 0.810990, 'Precision': 0.785212, 'Recall': 0.810990, 'F1 Score': 0.775461},
            'KNN': {'Accuracy': 0.806893, 'Precision': 0.781677, 'Recall': 0.806893, 'F1 Score': 0.784177},
            'Random Forest': {'Accuracy': 0.856340, 'Precision': 0.847805, 'Recall': 0.856340, 'F1 Score': 0.840945},
            'Naive Bayes': {'Accuracy': 0.626251, 'Precision': 0.750321, 'Recall': 0.626251, 'F1 Score': 0.660496},
            'Decision Tree': {'Accuracy': 0.795717, 'Precision': 0.794360, 'Recall': 0.795717, 'F1 Score': 0.795025},
            'SVM': {'Accuracy': 0.548774, 'Precision': 0.666799, 'Recall': 0.548774, 'F1 Score': 0.589193},
            'Neural Network': {'Accuracy': 0.857218, 'Precision': 0.848104, 'Recall': 0.857218, 'F1 Score': 0.843630}
        }
    },
    'Std': {
        'StandardScaler': {
            'LogisticRegression': {'Accuracy': 0.849201, 'Precision': 0.837873, 'Recall': 0.849201, 'F1 Score': 0.835378},
            'KNN': {'Accuracy': 0.809936, 'Precision': 0.784890, 'Recall': 0.809936, 'F1 Score': 0.785355},
            'Random Forest': {'Accuracy': 0.857218, 'Precision': 0.849292, 'Recall': 0.857218, 'F1 Score': 0.841408},
            'Naive Bayes': {'Accuracy': 0.659664, 'Precision': 0.772798, 'Recall': 0.659664, 'F1 Score': 0.690371},
            'Decision Tree': {'Accuracy': 0.798408, 'Precision': 0.796621, 'Recall': 0.798408, 'F1 Score': 0.797491},
            'SVM': {'Accuracy': 0.540348, 'Precision': 0.714948, 'Recall': 0.540348, 'F1 Score': 0.582478},
            'Neural Network': {'Accuracy': 0.854292, 'Precision': 0.844082, 'Recall': 0.854292, 'F1 Score': 0.843388}
        },
        'MinMaxScaler': {
            'LogisticRegression': {'Accuracy': 0.848675, 'Precision': 0.837216, 'Recall': 0.848675, 'F1 Score': 0.834763},
            'KNN': {'Accuracy': 0.818304, 'Precision': 0.797904, 'Recall': 0.818304, 'F1 Score': 0.799075},
            'Random Forest': {'Accuracy': 0.857511, 'Precision': 0.849710, 'Recall': 0.857511, 'F1 Score': 0.841691},
            'Naive Bayes': {'Accuracy': 0.659664, 'Precision': 0.772798, 'Recall': 0.659664, 'F1 Score': 0.690371},
            'Decision Tree': {'Accuracy': 0.798350, 'Precision': 0.796297, 'Recall': 0.798350, 'F1 Score': 0.797293},
            'SVM': {'Accuracy': 0.591375, 'Precision': 0.712348, 'Recall': 0.591375, 'F1 Score': 0.628476},
            'Neural Network': {'Accuracy': 0.858096, 'Precision': 0.848922, 'Recall': 0.858096, 'F1 Score': 0.845440}
        }
    }
}

# Create a function to extract data from the results
def create_dataframe(results):
    data = []
    for imputation, scalers in results.items():
        for scaler, models in scalers.items():
            for model, metrics in models.items():
                data.append({
                    'Imputation Method': imputation,
                    'Scaler': scaler,
                    'Model': model,
                    'Accuracy': metrics['Accuracy'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall'],
                    'F1 Score': metrics['F1 Score']
                })
    return pd.DataFrame(data)

# Create DataFrame
df = create_dataframe(results)
# Heatmap for F1 Score by scaler and imputation
heatmap_data = df.pivot_table(
    index='Imputation Method',
    columns='Scaler',
    values='F1 Score',
    aggfunc='mean'
)

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.3f')
plt.title('F1 Score by Scaler and Imputation Method', fontsize=16)
plt.ylabel('Imputation Method')
plt.xlabel('Scaler')
plt.tight_layout()
plt.show()

