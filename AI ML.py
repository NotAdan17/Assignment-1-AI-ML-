import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Global variables
df = None
model = None

def load_dataset():
    """
    Opens a file dialog to load a CSV or Excel file.
    Displays a message when the dataset is successfully loaded.
    """
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")])
    if file_path:
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, sep=None, engine='python')
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
            messagebox.showinfo("Success", f"Dataset loaded successfully!\nShape: {df.shape}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")

def train_model():
    """
    Trains a Random Forest classifier using user-specified features and target column.
    Displays model accuracy after training.
    """
    global model, df

    if df is None:
        messagebox.showerror("Error", "Please load a dataset first!")
        return

    try:
        features = [f.strip() for f in features_entry.get().split(',')]
        target = target_entry.get().strip()

        if not features or not target:
            messagebox.showerror("Error", "Please specify both features and target!")
            return

        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            messagebox.showerror("Error", f"Features not found in dataset: {missing_features}")
            return

        if target not in df.columns:
            messagebox.showerror("Error", f"Target column '{target}' not found in dataset!")
            return

        X = df[features]
        y = df[target]

        # Convert categorical columns to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes

        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        messagebox.showinfo("Model Trained", f"Model trained successfully! Accuracy: {accuracy:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {e}")

def make_predictions():
    """
    Makes predictions using the trained model on the entire dataset.
    Displays results in the text area.
    """
    global model, df

    if df is None:
        messagebox.showerror("Error", "Please load a dataset first!")
        return

    if model is None:
        messagebox.showerror("Error", "Please train the model first!")
        return

    try:
        features = [f.strip() for f in features_entry.get().split(',')]
        X_new = df[features]

        # Convert categorical columns to numeric (same as training)
        for col in X_new.columns:
            if X_new[col].dtype == 'object':
                X_new[col] = pd.Categorical(X_new[col]).codes

        # Handle missing values
        X_new = X_new.fillna(X_new.mean())

        predictions = model.predict(X_new)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Predictions:\n{predictions}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to make predictions: {e}")

# GUI Setup
root = tk.Tk()
root.title("Student Predictive Grades")

tk.Button(root, text="Load Dataset", command=load_dataset).pack(pady=10)

tk.Label(root, text="Features (comma-separated):").pack()
features_entry = tk.Entry(root, width=50)
features_entry.pack(pady=5)

tk.Label(root, text="Target:").pack()
target_entry = tk.Entry(root, width=50)
target_entry.pack(pady=5)

tk.Button(root, text="Train Model", command=train_model).pack(pady=10)
tk.Button(root, text="Make Predictions", command=make_predictions).pack(pady=10)

result_text = tk.Text(root, height=20, width=80)
result_text.pack(pady=10)

root.mainloop()
