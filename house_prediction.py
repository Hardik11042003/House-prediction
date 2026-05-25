import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# 1. Load a REAL dataset (California Housing)
print("Loading real housing data...")
california = fetch_california_housing()

# Create the DataFrame
df = pd.DataFrame(california.data, columns=california.feature_names)

# The Target (y) is the house price (expressed in hundreds of thousands of dollars)
y = pd.Series(california.target, name="Price_100k")

print("\n--- First 5 rows of our new features ---")
print(df.head())

# 2. Split the data (Notice X is now the entire DataFrame, not just one column)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

# 3. Train the model on multiple features
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions
predictions = model.predict(X_test)

# 5. Evaluate mathematically instead of visually
mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)

print("\n--- Model Evaluation ---")
# Prices are in $100,000s, so we multiply by 100,000 to see real dollar amounts
print(f"Mean Absolute Error: ${mae * 100000:,.0f} (On average, the model is off by this much)")
print(f"Root Mean Squared Error: ${rmse * 100000:,.0f}")

import matplotlib.pyplot as plt

print("\nGenerating graphs...")

# ==========================================
# GRAPH 1: Actual vs. Predicted Prices
# ==========================================
plt.figure(figsize=(10, 6))

# Plot the actual vs predicted prices as dots
# alpha=0.3 makes the dots slightly transparent so we can see where they overlap
plt.scatter(y_test, predictions, alpha=0.3, color='blue', label='AI Predictions')

# Draw the "Perfect Score" line
# If the AI was 100% accurate, every single blue dot would land exactly on this red line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Perfect Prediction Line')

plt.title('Actual vs. Predicted California House Prices')
plt.xlabel('Actual Price (in $100,000s)')
plt.ylabel('AI Predicted Price (in $100,000s)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==========================================
# GRAPH 2: Feature Importance (How the AI thinks)
# ==========================================
plt.figure(figsize=(10, 6))

# Extract the "importance" scores from the Random Forest model
# and wrap them in a Pandas Series for easy plotting
importances = pd.Series(model.feature_importances_, index=df.columns)

# Sort them from smallest to largest and plot as a horizontal bar chart
importances.sort_values().plot(kind='barh', color='teal')

plt.title('What Drives California House Prices the Most?')
plt.xlabel('Relative Importance (Scale 0.0 to 1.0)')
plt.ylabel('House Feature')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()