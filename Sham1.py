#!/usr/bin/env python
# coding: utf-8

# In[32]:


# Essential libraries for data manipulation, visualization, and analysis
import pandas as pd  # Data handling with DataFrames
import numpy as np  # Numerical operations
# Visualization libraries
import matplotlib.pyplot as plt  # Static visualizations
import seaborn as sns  # Advanced statistical plots
# Machine learning preprocessing and metrics
from sklearn.preprocessing import RobustScaler  # Scaling robust to outliers
from sklearn.metrics import silhouette_score  # Clustering quality evaluation
# Machine learning models
from sklearn.linear_model import LinearRegression  # Linear regression
from sklearn.cluster import KMeans  # K-Means clustering
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")


# In[33]:


data = pd.read_csv('business.retailsales.csv') # Load the dataset


# In[34]:


data.head()  # Displays the first 5 rows of the dataset


# In[35]:


data.info()  # Provides a summary of the DataFrame


# In[36]:


numeric_columns = data.select_dtypes(include=['number'])  # Filters columns with numeric data types


# In[37]:


# Calculating Statistical Moments
# These statistics provide valuable insights into the distribution and characteristics of the data.
stats_moments = pd.DataFrame({
    'Mean': numeric_columns.mean(),
    'Median': numeric_columns.median(),
    'Standard Deviation': numeric_columns.std(),
    'Skewness': numeric_columns.skew(),
    'Kurtosis': numeric_columns.kurt()})
stats_moments


# In[38]:


def draw_corr_heatmap(df):
    """
    GeneratesGenerates a correlation heatmap for numerical columns in the dataset. displaying triangle of correlations.
    """
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    
    # Mask the half of the correlation matrix
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create the heatmap with the mask applied
    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='coolwarm', fmt=".2f", cbar=True, vmin=-1, vmax=1)
    plt.title('Half Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.show()


# In[39]:


draw_corr_heatmap(numeric_columns)


# In[40]:


def plot_restaurant_distribution(df):
    """
    Creates a bar plot showing the count of each Restaurant ID.

    Parameters:
    - df : DataFrame : The dataset containing the data.
    """
    # Step 1: Count the occurrences of each Restaurant ID
    restaurant_counts = df['Product Type'].value_counts()

    # Step 2: Create the bar plot
    plt.figure(figsize=(6, 6))
    sns.barplot(
        x=restaurant_counts.index, 
        y=restaurant_counts.values, 
        palette="coolwarm"
    )
    
    # Step 3: Customize the plot
    plt.title("Distribution of Restaurant IDs", fontsize=16)
    plt.xlabel("Restaurant ID", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=90, fontsize=10, ha='right')  # Rotate x-axis labels for readability
    plt.tight_layout()
    
    # Step 4: Show the plot
    plt.show()


# In[41]:


plot_restaurant_distribution(data)


# In[42]:


def simple_linear_fit_with_predictions(df, independent_var, dependent_var, new_data=None):
    """
    Fits a simple linear regression model, plots the regression line with confidence intervals, 
    and optionally predicts values for new data points.
    
    Parameters:
    - df : DataFrame : The dataset containing the data.
    - independent_var : str : The name of the independent variable (feature).
    - dependent_var : str : The name of the dependent variable (target).
    - new_data : array-like, optional : New data points to predict. Default is None.
    
    Returns:
    - model : LinearRegression : The trained linear regression model.
    - predictions : array-like : Predictions for new data points (if provided).
    """
    # Step 1: Extract data for independent and dependent variables
    X = df[independent_var].values.reshape(-1, 1)  # Reshape to 2D for modeling
    y = df[dependent_var].values  # Dependent variable (target)

    # Step 2: Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Step 3: Generate predictions for the original data
    y_pred = model.predict(X)

    # Step 4: Predict values for new data points, if provided
    predictions = None
    if new_data is not None:
        new_data = np.array(new_data).reshape(-1, 1)
        predictions = model.predict(new_data)
        print("Predictions for new data points:")
        for i, pred in enumerate(predictions):
            print(f"  {independent_var} = {new_data[i][0]:.2f}, Predicted {dependent_var} = {pred:.2f}")

    # Step 5: Plot the regression line and data points
    plt.figure(figsize=(10, 6))
    sns.regplot(
        x=X.flatten(),
        y=y,
        ci=95,
        scatter_kws={'color': 'blue', 's': 50, 'alpha': 0.6},
        line_kws={'color': 'red', 'linewidth': 3, 'linestyle': '-'}
    )

    # Title and labels
    plt.title(f'Linear Regression of {dependent_var} vs {independent_var}', fontsize=16)
    plt.xlabel(independent_var, fontsize=14)
    plt.ylabel(dependent_var, fontsize=14)
    plt.legend(['Data points', 'Regression line', 'Confidence interval'], loc='upper left', fontsize=12)

    # Customize grid and layout
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    # Show the plot
    plt.show()

    return model, predictions


# In[43]:


new_data = [40,45,30]  # New x-values for prediction
model, new_y_pred = simple_linear_fit_with_predictions(data,'Net Quantity','Total Net Sales', new_data)


# In[44]:


data_for_clustering = data[['Net Quantity','Total Net Sales']].copy()

scaler = RobustScaler()
scaled_data = scaler.fit_transform(data_for_clustering)


# In[45]:


def compute_silhouette_and_inertia(cluster_count, data):
    """ 
    Calculates silhouette score and inertia for a given number of clusters.
    """
    kmeans_model = KMeans(n_clusters=cluster_count, n_init=20)
    kmeans_model.fit(data)  # Fit the model to the data
    labels = kmeans_model.labels_
    
    # Calculate silhouette score and inertia
    silhouette = silhouette_score(data, labels)
    inertia = kmeans_model.inertia_

    return silhouette, inertia

# Compute WCSS and silhouette scores
wcss_values = []
optimal_cluster_count, best_silhouette_score = None, -np.inf

# Loop through possible cluster counts and calculate silhouette score and inertia
for clusters in range(2, 11):  # Test from 2 to 10 clusters
    silhouette, inertia = compute_silhouette_and_inertia(clusters, scaled_data)
    wcss_values.append(inertia)
    
    # Update the best silhouette score and optimal cluster count
    if silhouette > best_silhouette_score:
        optimal_cluster_count = clusters
        best_silhouette_score = silhouette
        
    print(f"{clusters} clusters silhouette score = {silhouette:.2f}")

print(f"Optimal number of clusters = {optimal_cluster_count}")


# In[46]:


def plot_elbow_curve(min_clusters, max_clusters, wcss_values, optimal_clusters):
    """
    Plots the elbow curve to determine the best number of clusters (k).
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=144)
    
    # Plot WCSS values with blue line and diamond markers
    cluster_range = range(min_clusters, max_clusters + 1)
    ax.plot(cluster_range, wcss_values, marker='D', color='black', label='WCSS', markersize=8, linewidth=2)
    
    # Mark the optimal number of clusters with green cross
    ax.scatter(optimal_clusters, wcss_values[optimal_clusters - min_clusters], color='green', edgecolor='black', s=120, zorder=5, marker='X', label=f'Optimal k = {optimal_clusters}')
    ax.annotate(
        f'k={optimal_clusters}',
        xy=(optimal_clusters, wcss_values[optimal_clusters - min_clusters]),
        xytext=(optimal_clusters, wcss_values[optimal_clusters - min_clusters] + (max(wcss_values) - min(wcss_values)) * 0.05),
        fontsize=12,
        color='green',
        ha='center'
    )
    
    # Customize plot
    ax.set_xlabel('Number of Clusters (k)', fontsize=14)
    ax.set_ylabel('WCSS', fontsize=14)
    ax.set_title('Elbow Curve for Optimal k', fontsize=16)
    ax.set_xticks(cluster_range)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()

# Plot the elbow curve
plot_elbow_curve(2, 10, wcss_values, optimal_cluster_count)


# In[47]:


def perform_kmeans_clustering(original_data, normalized_data, scaler, cluster_range):
    """
    Performs K-Means clustering for a given range of k values and visualizes the results.
    Parameters: original_data (ndarray), normalized_data (ndarray), scaler (RobustScaler), cluster_range (iterable).
    """
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
        kmeans.fit(normalized_data)
        
        labels = kmeans.labels_
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        x_centers = centers[:, 0]
        y_centers = centers[:, 1]
        
        # Scatter plot with clusters
        plt.figure(figsize=(8, 6), dpi=144)
        scatter = plt.scatter(
            original_data[:, 0], original_data[:, 1], 
            c=labels, cmap='viridis', s=30, alpha=0.8, edgecolors='k', label='Data Points'
        )
        plt.scatter(
            x_centers, y_centers, 
            color='red', s=150, marker='x', edgecolor='black', label='Cluster Centers'
        )
        plt.title(f'K-Means Clustering with k={k}', fontsize=16)
        plt.xlabel('Net Quantity', fontsize=14)
        plt.ylabel('Total Net Sales', fontsize=14)
        plt.legend(fontsize=12)
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.show()

# Inverse normalization for accurate plotting of the original data
inverse_norm = scaler.inverse_transform(scaled_data)

# Perform clustering for the optimal k
perform_kmeans_clustering(inverse_norm, scaled_data, scaler, cluster_range=[optimal_cluster_count])


# In[ ]:





# In[ ]:





# In[ ]:




