import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("PRICIO – AI Pricing Intelligence")

st.write("Upload your weekly sales data to analyse optimal pricing.")

uploaded_file = st.file_uploader("Upload your sales CSV", type=["csv"])

if uploaded_file is None:

    st.info("Please upload a CSV file to begin analysis.")

else:

    data = pd.read_csv(uploaded_file)

    # ----------------------------
    # Load data
    # ----------------------------

    price = data["price"].values.reshape(-1,1)
    sales = data["units_sold"].values
    cost = data["cost_per_unit"].iloc[0]

    weeks = len(data)

    # ----------------------------
    # Remove Outliers
    # ----------------------------

    mean_sales = np.mean(sales)
    std_sales = np.std(sales)

    mask = (sales > mean_sales - 3*std_sales) & (sales < mean_sales + 3*std_sales)

    price = price[mask]
    sales = sales[mask]

    # ----------------------------
    # Smooth Sales
    # ----------------------------

    sales_smoothed = pd.Series(sales).rolling(window=3).mean().dropna()
    price_smoothed = price[len(price)-len(sales_smoothed):]

    # ----------------------------
    # Train Demand Model
    # ----------------------------

    model = LinearRegression()
    model.fit(price_smoothed, sales_smoothed)

    predicted_sales_training = model.predict(price_smoothed)

    # ----------------------------
    # Model Reliability (R²)
    # ----------------------------

    ss_res = np.sum((sales_smoothed - predicted_sales_training)**2)
    ss_tot = np.sum((sales_smoothed - np.mean(sales_smoothed))**2)

    r2 = 1 - (ss_res/ss_tot)

    # ----------------------------
    # Elasticity
    # ----------------------------

    slope = model.coef_[0]

    avg_price = np.mean(price_smoothed)
    avg_sales = np.mean(sales_smoothed)

    elasticity = slope * (avg_price / avg_sales)

    if elasticity < -1:
        sensitivity = "High"
    elif elasticity < -0.5:
        sensitivity = "Moderate"
    else:
        sensitivity = "Low"

    # ----------------------------
    # Model Confidence
    # ----------------------------

    if r2 > 0.6:
        confidence = "High"
    elif r2 > 0.3:
        confidence = "Medium"
    else:
        confidence = "Low"

    # ----------------------------
    # Generate Price Candidates
    # ----------------------------

    step = 5

    price_range = np.arange(avg_price*0.8, avg_price*1.2, step)

    profits = []
    predicted_sales_list = []

    for p in price_range:

        predicted_sales = model.predict([[p]])[0]

        predicted_sales_list.append(predicted_sales)

        profit = (p - cost) * predicted_sales

        profits.append(profit)

    # ----------------------------
    # Find Optimal Price Zone
    # ----------------------------

    max_profit = max(profits)

    best_prices = [p for p,profit in zip(price_range,profits) if profit >= 0.97*max_profit]

    min_price = round(min(best_prices))
    max_price = round(max(best_prices))

    # ----------------------------
    # Current Price Analysis
    # ----------------------------

    current_price = round(avg_price)

    current_sales = model.predict([[current_price]])[0]

    current_profit = (current_price - cost) * current_sales

    profit_gain = ((max_profit - current_profit)/current_profit)*100

    # ----------------------------
    # Diagnosis
    # ----------------------------

    if current_price < min_price:
        diagnosis = "Underpriced"
    elif current_price > max_price:
        diagnosis = "Potential Overpricing"
    else:
        diagnosis = "Near Optimal"

    # ----------------------------
    # Results
    # ----------------------------

    st.subheader("Pricing Diagnosis")

    st.write("Weeks Analysed:", weeks)

    st.write("Price Sensitivity:", sensitivity)
    st.write("Model Confidence:", confidence)
    st.write("Elasticity:", round(elasticity,2))

    st.write("Current Price:", current_price)

    st.write("Recommended Price Range:", f"{min_price} - {max_price}")

    st.write("Diagnosis:", diagnosis)

    st.write("Estimated Profit Improvement:", f"{round(abs(profit_gain))}%")

    # ----------------------------
    # Demand Curve
    # ----------------------------

    fig1, ax1 = plt.subplots()

    ax1.scatter(price_smoothed, sales_smoothed)
    ax1.plot(price_range, predicted_sales_list)

    ax1.set_title("Demand Curve")
    ax1.set_xlabel("Price")
    ax1.set_ylabel("Units Sold")

    st.pyplot(fig1)

    # ----------------------------
    # Profit Curve
    # ----------------------------

    fig2, ax2 = plt.subplots()

    ax2.plot(price_range, profits)

    ax2.set_title("Profit vs Price")
    ax2.set_xlabel("Price")
    ax2.set_ylabel("Profit")

    st.pyplot(fig2)
