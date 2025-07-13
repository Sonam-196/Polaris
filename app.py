import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from geopy.distance import geodesic

# Pincode → Coordinates + City mapping
pincode_coords = {
    '110001': (28.644800, 77.216721),  # Delhi
    '400001': (18.938771, 72.835335),  # Mumbai
    '560001': (12.9716, 77.5946),      # Bangalore
    '600001': (13.0827, 80.2707),      # Chennai
    '700001': (22.5726, 88.3639),      # Kolkata
    '122001': (28.4595, 77.0266),      # Gurgaon
    '500001': (17.3850, 78.4867),      # Hyderabad
}

pincode_city_map = {
    '110001': 'Delhi',
    '400001': 'Mumbai',
    '560001': 'Bangalore',
    '600001': 'Chennai',
    '700001': 'Kolkata',
    '122001': 'Gurgaon',
    '500001': 'Hyderabad',
}

# Estimate shipping days based on distance
def get_shipping_days(pin1, pin2):
    if pin1 in pincode_coords and pin2 in pincode_coords:
        dist = geodesic(pincode_coords[pin1], pincode_coords[pin2]).km
        return round(dist / 500 + 1)  # 500 km/day
    return 2

# Load & train model
@st.cache_data
def load_and_train_model():
    df = pd.read_csv("data_polaris.csv")
    df['TitleLength'] = df['Title'].apply(len)
    df['DescLength'] = df['Description'].apply(lambda x: len(x.split()))
    df['Visibility'] = df['PredictedVisibility'].map({'Low': 0, 'Medium': 1, 'High': 2})
    X = df[['TitleLength', 'DescLength', 'PriceStable', 'InStock', 'ShippingDays']]
    y = df['Visibility']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, df

# Load model and training data
model, data = load_and_train_model()

# --- UI ---
st.title("📦 Walmart Polaris Listing Visibility Predictor")
st.markdown("Enter product details and get smart visibility predictions based on Polaris ranking logic.")

title = st.text_input("Enter Product Title")
desc = st.text_area("Enter Product Description")
price_stable = st.radio("Is the Price Stable?", ["Yes", "No"])
in_stock = st.radio("Is the Product In Stock?", ["Yes", "No"])

seller_pin = st.selectbox("Enter Seller Pincode", options=list(pincode_city_map.keys()))
customer_pin = st.selectbox("Enter Customer Pincode", options=list(pincode_city_map.keys()))

# Show City Names
seller_city = pincode_city_map.get(seller_pin, "Unknown")
customer_city = pincode_city_map.get(customer_pin, "Unknown")
st.markdown(f"📍 **Seller Location**: {seller_city}")
st.markdown(f"📍 **Customer Location**: {customer_city}")

# Predict Button
if st.button("Predict Visibility"):
    title_len = len(title)
    desc_len = len(desc.split())
    price_flag = 1 if price_stable == "Yes" else 0
    stock_flag = 1 if in_stock == "Yes" else 0
    shipping_days = get_shipping_days(seller_pin, customer_pin)

    input_data = [[title_len, desc_len, price_flag, stock_flag, shipping_days]]
    pred = model.predict(input_data)[0]
    rev_map = {0: 'Low', 1: 'Medium', 2: 'High'}

    st.success(f"📈 Predicted Visibility: **{rev_map[pred]}**")
    st.write(f"🚚 Estimated Shipping Days: **{shipping_days}**")

    # --- Suggestions ---
    st.subheader("💡 Suggestions to Improve Visibility")
    suggestions = []
    avg_title = data['TitleLength'].mean()
    avg_desc = data['DescLength'].mean()
    avg_shipping = data['ShippingDays'].mean()

    if title_len < avg_title:
        suggestions.append(f"🔹 Increase title length. Current: {title_len}, Avg: {int(avg_title)}")
    if desc_len < avg_desc:
        suggestions.append(f"🔹 Add more details/keywords. Current: {desc_len}, Avg: {int(avg_desc)}")
    if price_flag == 0:
        suggestions.append("🔹 Stabilize your pricing.")
    if stock_flag == 0:
        suggestions.append("🔹 Ensure product is in stock.")
    if shipping_days > avg_shipping:
        suggestions.append(f"🔹 Try reducing shipping time. Avg recommended: ≤ {int(avg_shipping)} days")

    if suggestions:
        for s in suggestions:
            st.write(s)
    else:
        st.info("✅ Your listing already meets most best practices. Great job!")
