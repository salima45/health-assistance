import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
data = pd.read_csv('disease.csv')

# Input and output
X = data[['fever', 'cough', 'headache', 'fatigue']]
y = data['disease']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model Trained Successfully")