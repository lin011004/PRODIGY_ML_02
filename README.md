# Mall Customer Segmentation

This project involves segmenting customers based on their annual income and spending score using K-means clustering. The dataset used is the Mall Customers dataset, which contains information about customers' age, annual income, and spending score.

## Project Structure

The project consists of the following files:

- `Mall_Customers.csv`: The dataset containing customer information.
- `Task2.py`: The Python script containing the code for preprocessing, clustering, and visualizing the customer segments.
- `README.md`: This file.

## Requirements

- Python 3.6+
- pandas
- scikit-learn
- matplotlib

You can install the required libraries using pip:

```sh
pip install pandas scikit-learn matplotlib
```

## Code Overview

The script `Task2.py` performs the following steps:

1. **Load the Data**: The data is loaded from `Mall_Customers.csv`.

    ```python
    data = pd.read_csv("C:\\Users\\Lingesh\\OneDrive\\Desktop\\ML Intern\\archive\\Mall_Customers.csv")
    ```

2. **Data Preprocessing**: Handle missing values and scale numerical features.
    - Handle missing values using the mean strategy with `SimpleImputer`.

        ```python
        imputer = SimpleImputer(strategy='mean')
        data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] = imputer.fit_transform(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
        ```

    - Scale the features `Annual Income (k$)` and `Spending Score (1-100)` using `StandardScaler`.

        ```python
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[['Annual Income (k$)', 'Spending Score (1-100)']])
        ```

3. **Select Features**: Select the scaled features for clustering.

    ```python
    X = scaled_data
    ```

4. **Choose the Number of Clusters**: Determine the optimal number of clusters using the elbow method.

    ```python
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()
    ```

5. **Apply K-means Clustering**: Apply K-means clustering with the chosen number of clusters (k = 5).

    ```python
    k = 5  # Choose the number of clusters based on elbow method
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(X)
    ```

6. **Visualize the Clusters**: Visualize the clusters along with the centroids.

    ```python
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)  # Plot data points with cluster colors
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')  # Plot centroids
    plt.title('Clusters of Customers')
    plt.xlabel('Annual Income (scaled)')
    plt.ylabel('Spending Score (scaled)')
    plt.legend()
    plt.grid(True)
    plt.show()
    ```

7. **Interpret the Clusters**: Add the cluster labels to the original data and compute the mean of each cluster.

    ```python
    data['cluster'] = clusters
    # Exclude non-numeric columns before computing the mean
    numeric_data = data.select_dtypes(include='number')
    cluster_means = numeric_data.groupby('cluster').mean()
    print(cluster_means)
    ```

## Running the Code

1. Ensure that `Mall_Customers.csv` is placed in the appropriate directory as specified in the script.
2. Run the script `Task2.py`.

    ```sh
    python Task2.py
    ```

3. The script will display the elbow method plot, visualize the clusters, and print the mean values for each cluster.

## Acknowledgements

This project uses the Mall Customers dataset available on [Kaggle](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python).



---

