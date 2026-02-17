# Sea-Ice-Classification-with-Unsupervised-Learning

This project investigates the use of unsupervised machine learning techniques to distinguish between sea ice and leads in satellite observations. Using Sentinel altimetry data, the code classifies surface types based on their measured characteristics without relying on predefined labels, and then compares the resulting classifications with the classification provided by the European Space Agency (ESA).

Two approaches are examined: K-means clustering and Gaussian Mixture Models. K-means separates observations into clusters based on geometric similarity, while GMMs provide a probabilistic framework that allows overlapping and non-spherical structures to be represented. By applying and comparing these methods, the project evaluates how different modelling assumptions affect the identification of sea ice and leads and assesses their suitability for remote sensing analysis.

# Aims

The tasks in this notebook will be mainly two:

1) Discrimination of Sea ice and lead based on image classification based on Sentinel-2 optical data.
2) Discrimination of Sea ice and lead based on altimetry data classification based on Sentinel-3 altimetry data.

# Getting Started: Installation and Prerequisites

## Google Drive

Google Drive must be mounted in Google Colab using the following code:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## pip Packages

Secondly, the following packages must be installed to conduct the project:

```python
pip install rasterio
pip install netCDF4
```

## Sentinel Datasets

Metadata for Sentinel-2 and Sentinel-3 OLCI must be downloaded and uploaded to Google Drive before the Unsupervised Learning code can be run. It should be uploaded to the same folder the Google Colab file is stored in. This data is provided in 2 folders with the following names:

- S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE
- S3A_SR_2_LAN_SI_20190307T005808_20190307T012503_20230527T225016_1614_042_131______LN3_R_NT_005.SEN3


# K-Means Clustering

## Introduction to K-means Clustering
K-means clustering is a type of unsupervised learning algorithm used for partitioning a dataset into a set of k groups (or clusters), where k represents the number of groups pre-specified by the analyst. It classifies the data points based on the similarity of the features of the data [MacQueen and others, 1967]. The basic idea is to define k centroids, one for each cluster, and then assign each data point to the nearest centroid, while keeping the centroids as small as possible.

## Why K-means for Clustering?
K-means clustering is particularly well-suited for applications where:

- The structure of the data is not known beforehand: K-means doesn’t require any prior knowledge about the data distribution or structure, making it ideal for exploratory data analysis.
- Simplicity and scalability: The algorithm is straightforward to implement and can scale to large datasets relatively easily.

## Key Components of K-means
- Choosing K: The number of clusters (k) is a parameter that needs to be specified before applying the algorithm.
- Centroids Initialization: The initial placement of the centroids can affect the final results.
- Assignment Step: Each data point is assigned to its nearest centroid, based on the squared Euclidean distance.
- Update Step: The centroids are recomputed as the center of all the data points assigned to the respective cluster.

## The Iterative Process of K-means
The assignment and update steps are repeated iteratively until the centroids no longer move significantly, meaning the within-cluster variation is minimised. This iterative process ensures that the algorithm converges to a result, which might be a local optimum.

## Advantages of K-means
- Efficiency: K-means is computationally efficient.
- Ease of interpretation: The results of k-means clustering are easy to understand and interpret.

# Gaussian Mixture Models
Gaussian Mixture Models (GMM) are a probabilistic model for representing normally distributed subpopulations within an overall population. The model assumes that the data is generated from a mixture of several Gaussian distributions, each with its own mean and variance [Reynolds and others, 2009]. GMMs are widely used for clustering and density estimation, as they provide a method for representing complex distributions through the combination of simpler ones.

## Why Gaussian Mixture Models for Clustering?
Gaussian Mixture Models are particularly powerful in scenarios where:

- Soft clustering is needed: Unlike K-means, GMM provides the probability of each data point belonging to each cluster, offering a soft classification and understanding of the uncertainties in our data.
- Flexibility in cluster covariance: GMM allows for clusters to have different sizes and different shapes, making it more flexible to capture the true variance in the data.

## Key Components of GMM
- Number of Components (Gaussians): Similar to K in K-means, the number of Gaussians (components) is a parameter that needs to be set.
- Expectation-Maximization (EM) Algorithm: GMMs use the EM algorithm for fitting, iteratively improving the likelihood of the data given the model.
- Covariance Type: The shape, size, and orientation of the clusters are determined by the covariance type of the Gaussians (e.g., spherical, diagonal, tied, or full covariance).

## The EM Algorithm in GMM
The Expectation-Maximization (EM) algorithm is a two-step process:

1) Expectation Step (E-step): Calculate the probability that each data point belongs to each cluster.
2) Maximization Step (M-step): Update the parameters of the Gaussians (mean, covariance, and mixing coefficient) to maximize the likelihood of the data given these assignments.

This process is repeated until convergence, meaning the parameters do not significantly change from one iteration to the next.

## Advantages of GMM
- Soft Clustering: Provides a probabilistic framework for soft clustering, giving more information about the uncertainties in the data assignments.
- Cluster Shape Flexibility: Can adapt to ellipsoidal cluster shapes, thanks to the flexible covariance structure.

# Further Reading

The following sources are useful for understanding unsupervised learning:

- Chapter 9 "Mixture Models and EM" (pgs. 423-455) *Pattern Recognition and Machine Learning* by Christopher M. Bishop
- Chapter 12 "Unsupervised Learning" (pgs. 503-556) *An Introduction to Statistical Learning with Applications in Python* by Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor
- Scikit User Guide available at: https://scikit-learn.org/stable/user_guide.html

# Acknowledgements

Assignment 4 completed as a part of GEOL0069 at University College London.

# References

Christopher M Bishop and Nasser M Nasrabadi. *Pattern recognition and machine learning*. Volume 4. Springer, 2006.

Douglas A Reynolds and others. Gaussian mixture models. *Encyclopedia of biometrics*, 2009.

James MacQueen and others. Some methods for classification and analysis of multivariate observations. *In Proceedings of the fifth Berkeley symposium on mathematical statistics and probability*, volume 1, 281–297. Oakland, CA, USA, 1967.

# Contact

Email: ishita.chauhan.23@ucl.ac.uk / ishitachn@gmail.com

Project link (shareable): https://github.com/typhlosion-0/Sea-Ice-Classification-with-Unsupervised-Learning/tree/main

