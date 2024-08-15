import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import classification_report


df_train = pd.read_csv("C:\\Users\\X1 Carbon\\Downloads\\archive (2)\\UNSW_NB15_testing-set.csv")
df_test = pd.read_csv("C:\\Users\\X1 Carbon\\Downloads\\archive (2)\\UNSW_NB15_training-set.csv")
print("Length of training set: ", len(df_train))
print("Length of testing set: ", len(df_test))

df = pd.concat([df_train, df_test])
df = df.drop(columns=['id', 'label', 'sloss', 'dloss', 'dwin', 'ct_ftp_cmd'])
df_cat = ['proto', 'service', 'state']
print(df_cat)
for feature in df_cat:
    df[feature] = LabelEncoder().fit_transform(df[feature])



X = df.drop(columns=['attack_cat'])
feature_list = list(X.columns)
X = np.array(X)
y = df['attack_cat']


print(X[0])
print(feature_list)

from imblearn.over_sampling import SMOTE
# Specify desired class ratios for 'dos', 'analysis', and 'backdoor' classes
class_ratios = {
    'DoS':17000 ,       # Resample 'dos' class to 50% of its original size
    'Analysis': 11000,  # Resample 'analysis' class to 80% of its original size
    'Backdoor': 11000   # Keep 'backdoor' class unchanged (100% of its original size)
}

# Initialize SMOTE sampler
smote = SMOTE(sampling_strategy=class_ratios)


X, y = smote.fit_resample(X, y)

smote = SMOTE(sampling_strategy='all')

X, y = smote.fit_resample(X, y)
y = pd.Series(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the scaler on X_train
X_train_scaled = scaler.fit_transform(X_train)

# Transform X_test using the fitted scaler from X_train
X_test_scaled = scaler.transform(X_test)
print(X_test[0])
print(X_test_scaled[0])
# Initialize dictionaries to store metrics
train_score = {}
accuracy = {}
precision = {}
recall = {}
training_time = {}
y_pred = {}


rfc_model = RandomForestClassifier(n_estimators=30, max_depth=30, min_samples_split=4, min_samples_leaf=2)
start_time = time.time()
rfc_model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

# Make predictions
y_pred = rfc_model.predict(X_test_scaled)

# Calculate metrics
train_score = rfc_model.score(X_train_scaled, y_train)
accuracy = rfc_model.score(X_test_scaled, y_test)
precision = precision_score(y_test, y_pred, average='weighted')
recall= recall_score(y_test, y_pred, average='weighted')


# Save the trained model
joblib.dump(rfc_model, 'random_forest_classifier_model(12).pkl')

joblib.dump(scaler, 'scaler4.pkl')
print(accuracy)
print(precision)
print(recall)



# Generate classification report
report_xgb = classification_report(y_test,y_pred)

# Print classification report
print(report_xgb)


