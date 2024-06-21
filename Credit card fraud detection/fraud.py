import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Load your data
df = pd.read_csv('creditcard.csv')

# Then I Preprocess and normalize data
df['normalizedAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df = df.drop(['Amount'],axis=1)

# Handling class imbalance
smote = SMOTE(random_state=0)
X_resampled, y_resampled = smote.fit_resample(df.drop('Class', axis=1), df['Class'])

# Split the data into test and train datasets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

# Train the model with LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# To Make predictions with the model
y_pred = model.predict(X_test)

# This is a hypothetical example and assumes that I have a trained model and a preprocessed transaction

# Let's say we have a new transaction with the following features
new_transaction = [0.0, -2.3122265423263, 1.95199201064158, -1.60985073229769, 3.9979055875468, -0.522187864667764,
                   -1.42654531920595, -2.53738730624579, 1.39165724829804, -2.77008927719433, -2.77227214465915,
                   3.20203320709635, -2.89990738849473, -0.595221881324605, -4.28925378244217, 0.389724120274487,
                   -1.14074717980657, -2.83005567450437, -0.0168224681808257, 0.416955705037907, 0.126910559061474,
                   0.517232370861764, -0.0350493686052974, -0.465211076182388, 0.320198198514526, 0.0445191674731724,
                   0.177839798284401, 0.261145002567677, -0.143275874698919, 0.01]

new_transaction = pd.DataFrame([new_transaction], columns=df.columns.drop('Class'))

# Then I convert the transaction to a numpy array and reshape it because the model expects a 2D array

new_transaction = np.array(new_transaction).reshape(1, -1)

# Use the model to predict the class of the new transaction
fraud_prediction = model.predict(new_transaction)

if fraud_prediction == 0:
    print("The transaction is genuine.")
else:
    print("The transaction is fraudulent.")


# I want to evaluate the model with precision score, recall score and f1 score.
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1-score: {f1_score(y_test, y_pred)}')
