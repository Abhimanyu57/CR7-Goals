# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("data.csv")

# Drop unnecessary columns
df = df.drop(columns=["Playing_Position", "Goal_assist"], errors="ignore")

# Create target variable
def map_result(result):
    try:
        home, away = map(int, result.split(":"))
        if home > away:
            return "Win"
        elif home < away:
            return "Loss"
        else:
            return "Draw"
    except:
        return None

df["Target"] = df["Result"].apply(map_result)
df = df.dropna(subset=["Target"])

# Separate features and target
X = df.drop(columns=["Result", "Target"])
y = df["Target"]

# Identify column types
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns

# Common preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols),
    ]
)

# Models
log_reg = LogisticRegression(max_iter=1000)
tree = DecisionTreeClassifier(max_depth=6, random_state=42)

log_reg_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", log_reg),
    ]
)

tree_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", tree),
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train and evaluate Logistic Regression
log_reg_pipeline.fit(X_train, y_train)
log_pred = log_reg_pipeline.predict(X_test)

print("Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, log_pred))
print(classification_report(y_test, log_pred))

# ----------------------------
# Train and evaluate Decision Tree
# ----------------------------
tree_pipeline.fit(X_train, y_train)
tree_pred = tree_pipeline.predict(X_test)

print("\nDecision Tree Results")
print("Accuracy:", accuracy_score(y_test, tree_pred))
print(classification_report(y_test, tree_pred))
