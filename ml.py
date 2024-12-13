import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = r"C:\Users\DELL.DESKTOP-F997LTH\Downloads\Copper_Set (1).xlsx"  # Use raw string for file path
data = pd.ExcelFile(file_path)
df = data.parse('Result 1')

# Data cleaning and preprocessing
df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
df['item_date'] = pd.to_datetime(df['item_date'], errors='coerce')
df['delivery date'] = pd.to_datetime(df['delivery date'], errors='coerce')
df['material_ref'] = df['material_ref'].replace(to_replace=r'^00000.*', value=None, regex=True)
df.fillna({
    'country': df['country'].mode()[0],
    'application': df['application'].mode()[0],
    'thickness': df['thickness'].median(),
    'width': df['width'].median(),
    'material_ref': 'Unknown'
}, inplace=True)
df.dropna(subset=['selling_price', 'status'], inplace=True)

label_encoders = {}
for col in ['country', 'application', 'material_ref', 'product_ref', 'item type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

regression_target = 'selling_price'
X = df.drop(columns=[regression_target, 'status', 'id', 'item_date', 'delivery date', 'customer'])
y_reg = df[regression_target]

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_imputed, y_reg, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

pca = PCA(n_components=0.95)
X_train_reg_pca = pca.fit_transform(X_train_reg_scaled)
X_test_reg_pca = pca.transform(X_test_reg_scaled)

regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train_reg_pca, y_train_reg)

joblib.dump(regressor, 'copper_price_predictor_model.pkl')

model = joblib.load('copper_price_predictor_model.pkl')

# Streamlit app
st.title("Copper Price Prediction")
st.write("Enter the following parameters to predict the copper price:")

quantity_tons_log = st.number_input("Quantity Tons (Log)", value=0.0)
customer = st.number_input("Customer", value=0)
country = st.number_input("Country", value=0)
status = st.number_input("Status", value=0)
application = st.number_input("Application", value=0)
width = st.number_input("Width", value=0)
product_ref = st.number_input("Product Ref", value=0)
thickness_log = st.number_input("Thickness (Log)", value=0.0)
selling_price_log = st.number_input("Selling Price (Log)", value=0.0)
item_date_day = st.number_input("Item Date Day", value=1)
item_date_month = st.number_input("Item Date Month", value=1)
item_date_year = st.number_input("Item Date Year", value=2021)

if st.button("Predict"):
    user_data = np.array([[quantity_tons_log, customer, country, status, application,
                           width, product_ref, thickness_log, selling_price_log,
                           item_date_day, item_date_month, item_date_year]])

    user_data_imputed = imputer.transform(user_data)
    user_data_scaled = scaler.transform(user_data_imputed)
    user_data_pca = pca.transform(user_data_scaled)

    prediction = model.predict(user_data_pca)[0]

    st.write(f"Predicted Copper Price: {prediction}")
