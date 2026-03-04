import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

print("\n------ PRICIO PRICING ANALYSIS ------\n")

# ----------------------------
# Load data
# ----------------------------

data = pd.read_csv("sales_data.csv")

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
# Calculate R² (model reliability)
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

# ----------------------------
# Elasticity interpretation
# ----------------------------

if elasticity < -1:
    sensitivity = "High"
elif elasticity < -0.5:
    sensitivity = "Moderate"
else:
    sensitivity = "Low"

# ----------------------------
# Model confidence
# ----------------------------

if r2 > 0.6:
    confidence = "High"
elif r2 > 0.3:
    confidence = "Medium"
else:
    confidence = "Low"

# ----------------------------
# Generate price candidates
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
# Find optimal price zone
# ----------------------------

max_profit = max(profits)

best_prices = [p for p,profit in zip(price_range,profits) if profit >= 0.97*max_profit]

min_price = round(min(best_prices))
max_price = round(max(best_prices))

# ----------------------------
# Current price analysis
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

print("Data analysed:", weeks, "weeks\n")

print("Price Sensitivity:", sensitivity)
print("Model Confidence:", confidence)
print("Elasticity:", round(elasticity,2), "\n")

print("Current Price:", current_price)

print("Recommended Price Range:", min_price, "-", max_price)

print("Diagnosis:", diagnosis)

print("Estimated Profit Improvement:", round(abs(profit_gain)), "%")

# ----------------------------
# Demand Curve
# ----------------------------

plt.figure()

plt.scatter(price_smoothed, sales_smoothed)

plt.plot(price_range, predicted_sales_list)

plt.title("Demand Curve")

plt.xlabel("Price")

plt.ylabel("Units Sold")

plt.show()

# ----------------------------
# Profit Curve
# ----------------------------

plt.figure()

plt.plot(price_range, profits)

plt.title("Profit vs Price")

plt.xlabel("Price")

plt.ylabel("Profit")

plt.show()
