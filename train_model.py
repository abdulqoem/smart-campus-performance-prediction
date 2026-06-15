import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
df = pd.read_csv("Student_Performance_Dataset.csv")

print("Dataset shape:", df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())

# 2. Choose target
target = "Performance_Level"

# 3. Drop columns not needed
# Final_Percentage is dropped because it directly determines Performance_Level
drop_cols = ["Student_ID", "Final_Percentage", "Pass_Fail"]

df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# 4. Handle missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# 5. Encode categorical columns
encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# 6. Split X and y
X = df.drop(columns=[target])
y = df[target]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Model 1: Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("\nDecision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

# 8. Model 2: Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# 9. Save best model
joblib.dump(rf_model, "student_performance_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(feature_names, "feature_names.pkl")

print("\nModel saved successfully.")