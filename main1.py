import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from tqdm import tqdm

train_data = pd.read_csv("D:\\coding stuff\\ai&ml\\genderpridict\\genderpridict\\archive\\Training set.csv")
test_data = pd.read_csv("D:\\coding stuff\\ai&ml\\genderpridict\\genderpridict\\archive\\Test set.csv")
encoder = LabelEncoder()
train_data['Sex'] = encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = encoder.transform(test_data['Sex']) 
train_data['BMI'] = train_data['Weight'] / ((train_data['Height'] / 100) ** 2)
train_data['Height_Weight_Ratio'] = train_data['Height'] / train_data['Weight']

test_data['BMI'] = test_data['Weight'] / ((test_data['Height'] / 100) ** 2)
test_data['Height_Weight_Ratio'] = test_data['Height'] / test_data['Weight']
X_train = train_data[['Height', 'Weight', 'BMI', 'Height_Weight_Ratio']]
y_train = train_data['Sex']

X_test = test_data[['Height', 'Weight', 'BMI', 'Height_Weight_Ratio']]
y_test = test_data['Sex']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'class_weight': ['balanced', None]
}

tqdm.pandas()

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, 
                           cv=5, verbose=0, n_jobs=-1, scoring='accuracy')

with tqdm(total=1, desc="Training Model", ncols=100) as pbar:
    grid_search.fit(X_train_scaled, y_train)
    pbar.update(1)

best_model = grid_search.best_estimator_

joblib.dump(best_model, 'gender_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cross_val_scores}")
print(f"Average CV score: {cross_val_scores.mean():.2f}")
