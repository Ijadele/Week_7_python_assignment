from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Iris dataset and save as CSV
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df.to_csv('iris_dataset.csv', index=False)
print("Dataset saved as iris_dataset.csv")

# Task 1: Load and Explore the Dataset
try:
    # Load the dataset
    file_path = "iris_dataset.csv"
    data = pd.read_csv(file_path)
    
    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(data.head())
    
    # Check the structure of the dataset
    print("\nDataset Info:")
    print(data.info())
    
    # Check for missing values
    print("\nMissing values:")
    print(data.isnull().sum())
    
    # Fill or drop missing values
    data = data.fillna(data.mean())
    print("\nAfter handling missing values:")
    print(data.isnull().sum())

except FileNotFoundError:
    print("The file was not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")

# Task 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic Statistics:")
print(data.describe())

# Perform grouping and compute mean for each group
grouped_data = data.groupby('species').mean()
print("\nGrouped Data Mean:")
print(grouped_data)

# Task 3: Data Visualization
# Customize Seaborn styles
sns.set(style="whitegrid")

# 1. Line Chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x=data.index, y='sepal length (cm)', hue='species')
plt.title("Line Chart: Sepal Length Trend Across Species")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.show()

# 2. Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='species', y='sepal width (cm)')
plt.title("Bar Chart: Average Sepal Width per Species")
plt.xlabel("Species")
plt.ylabel("Sepal Width (cm)")
plt.show()

# 3. Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data['petal length (cm)'], bins=30, kde=True)
plt.title("Histogram: Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Scatter Plot: Sepal vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

# Findings and Observations
print("\nFindings and Observations:")
print("- Sepal length tends to increase with petal length across all species.")
print("- Species 2 shows a higher average for petal length compared to others.")
