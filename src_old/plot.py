"""
    Description: This file contains:
        - methods to calculate the optimal number of clusters using the elbow method and silhouette method
        - the code to plot the data

    Author : Gallegos Carvajal Ian Marco
"""

import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
from yellowbrick.cluster import KElbowVisualizer

from utils import vis_save_image_pdf, plt_save_image_pdf


def elbow_method(df, feature_emb, image_out):
    matrix = np.vstack(df[feature_emb].values)
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, 50))
    visualizer.fit(matrix)
    visualizer.show()
    # save the plot
    vis_save_image_pdf(visualizer, image_out + '_' + 'elbow_method')

    # return the optimal number of clusters
    print(f" Optimal number of clusters: {visualizer.elbow_value_}")
    return visualizer.elbow_value_


def silhouette_method(df, feature_emb, image_out):
    matrix = np.vstack(df[feature_emb].values)
    sse = []
    K = range(2, 50)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(matrix)
        sse.append(kmeans.inertia_)  # inertia_ is l'SSE for K-means

    # visualize graphic SSE vs k
    plt.figure(figsize=(8, 6))
    plt.plot(K, sse, 'bx-')
    plt.xlabel('Number of cluster k')
    plt.ylabel('Sum of SSE')
    plt.title('Elbow method for optimal k')
    plt.show()
    plt_save_image_pdf(plt, image_out + '_' + 'silhouette_method')


def plot_2d(df, feature, matrix_origin, n_clusters, image_out):
    tsne = TSNE(n_components=2, random_state=42)
    matrix = tsne.fit_transform(matrix_origin)
    df['X'] = matrix[:, 0]
    df['Y'] = matrix[:, 1]

    plt.figure(figsize=(20, 20))
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_clusters))
    for i, color in enumerate(colors):
        plt.scatter(matrix[df['Cluster'] == i, 0], matrix[df['Cluster'] == i, 1], c=[color], label='Cluster ' + str(i))
    plt.title('Clusters of ' + feature + 'with' + str(n_clusters))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    plt_save_image_pdf(plt, image_out + '_' + 'cluster' + '_' + '2d')


def plot_3d(df, feature, matrix_origin, n_clusters, image_out):
    tsne = TSNE(n_components=3, random_state=42)
    matrix = tsne.fit_transform(matrix_origin)
    df['X'] = matrix[:, 0]
    df['Y'] = matrix[:, 1]
    df['Z'] = matrix[:, 2]

    for t in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, n_clusters))
        for i, color in enumerate(colors):
            ax.scatter(matrix[df['Cluster'] == i, t[0]], matrix[df['Cluster'] == i, t[1]], matrix[df['Cluster'] == i, t[2]], c=[color], label='Cluster ' + str(i), s=10)
        plt.title('Clusters of ' + feature + 'with' + str(n_clusters))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
        plt_save_image_pdf(plt, image_out + '_' + 'cluster' + '_' + '3d' '_' + str(t[0]) + '_' + str(t[1]) + '_' + str(t[2]))

def bar_plot(df, feature, image_out):
    print("Plotting bar plot of " + feature + "...")
    df[feature].value_counts().plot(kind='bar')
    plt.title('Bar plot of ' + feature)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()
    plt_save_image_pdf(plt, image_out + '_' + 'bar_plot')

def visualize_clusters(df, feature, feature_emb, n_clusters, image_out):
    matrix = np.vstack(df[feature_emb].values)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(matrix)
    df['Cluster'] = kmeans.labels_

    # save the csv with the cluster
    df.to_csv('results/' + image_out + '_' + 'cluster.csv', index=False)

    plot_2d(df, feature, matrix, n_clusters, image_out)
    plot_3d(df, feature, matrix, n_clusters, image_out)



