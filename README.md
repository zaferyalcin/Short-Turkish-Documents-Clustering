# Short-Turkish-Documents-Clustering
  In this project, short document clustering algorithms for the English language were compared.

## DATASET
  The dataset we will use for the Turkish dataset will be the Turkish News Category
Dataset compiled from print media and news sites published by Interpress Media
Monitoring Company between 2010-2017 to be used for machine learning. TNCD; It
consists of 4900 news gathered under 7 categories: siyaset, dunya, ekonomi, kultur, saglik,
spor, teknoloji.

## DATA PREPARATIONS
  Data preparation is the process of cleaning and transforming raw data prior to
processing and analysis. It is an important step prior to processing and often involves
reformatting data, making corrections to data and the combining of data sets to enrich data.

  While a lot of low-quality information is available in various data sources and on the Web,
many organizations or companies are interested in how to transform the data into cleaned
forms which can be used for high-profit purposes.

  In NLP, text preprocessing is the first step in the process of building a model.
Applied text preprocessing steps are:

### Normalization:
  Text data contains a lot of noise, this takes the form of special characters such as
hashtags, punctuation and numbers. All of which are difficult for computers to understand if
they are present in the data. Normalization is the process of the data to remove these
elements.

### Tokenization:
  Splitting the data into sentences, words, letters or n-grams. There are lots of different
techniques for tokenization. A few of them are White Space Tokenization, Dictionary based
Tokenization, Rule based Tokenization, Penn Tree Tokenization.

  Tokenization example with White Space technique on the given sentence;
  (A small meteorite crashed into a wooded area in Nicaragua) = [‘A’,’small’,’meteorite’,’crashed’,’into’,’a’,’wooded’,’area’,’in’,’Nicaragua’
  
### Lower-casing:
  Converting a word to lowercase. For example (WORD > word).
Words like Answer and ansWeR mean the same but when not converted to the lower case
those two are represented as two different words in the vector space model. Usually we
do not want this situation to happen. That’s why we convert to lowercase.

### Handling stop words:
  Stop words are commonly occurring words that for some computational processes
provide little information or in some cases introduce unnecessary noise and therefore need to
be removed. This is particularly the case for text classification tasks.

### Stemming:
  Stemming is the process of reducing words to their root form. For example, the
words “work”, “working” and “worked” have very similar, and in many cases, the same
meaning.
  The process of stemming will reduce these to the root form of “work”. This is again a way to
reduce noise and the dimensionality of the data. Stemming usually refers to a crude heuristic
process that chops off the ends of words in the hope of achieving this goal correctly most of
the time, and often includes the removal of derivational affixes.

### Lemmatization:
  The goal of lemmatization is the same as for stemming, in that it aims to reduce
words to their root form. It usually refers to doing things properly with the use of a
vocabulary and morphological analysis to more accurately find the root, or “lemma” for a
word

## ALGORITHMS
  Many algorithms available for clustering topics. These topic can be categorized
several fields such as methods the following:

  ● partitioning methods
  
  ● hierarchical methods
  
  ● density-based methods
  
  ● grid-based methods
  
  
  In the term project, three clustering algorithms must be used and then compared to
each other

### K-Means
  The algorithm to be used in the partitioning methods is the k-means algorithm. The
K-means algorithm is the most popular and simplest ML algorithm between clustering topics.
We can implement the K-means algorithm according to the following pseudocode:
  ● The number of clusters desired to be formed is taken from the user as the k value.
  
  ● At the end of the algorithm, k points, which we expect to be the center, are
    randomly added to the data space.
    
  ● The borders of randomly assigned centers are determined to be separated from
    each other, provided that no data will be left out.
    
  ● Operations based on mean and euclidean distance are applied to each of the data
    and then enters and borders are relocated.
    
  ● These processes are taken recursively until the location of the centers and borders
    remains the same as before the last process.

### DBSCAN Algorithm
  The DBSCAN algorithm is based on revealing the neighborhood of data points in two or
multidimensional spaces. It refers to unsupervised learning methods that identify different
groups/clusters in the data, based on the idea that a cluster in the data space is a
contiguous region with high point density and is separated from other such clusters by
contiguous regions with low point density. The database is mostly used in the analysis of
spatial data because it deals with spatial perspective.

  Core, Eps, MinPts, density reachable point, density connected point terms are the basic
concepts for DBSCAN algorithm. The algorithm takes the maximum radius of the
neighborhood (Eps) and minimum number of points in an Eps-neighborhood of that point
(MinPts) as input parameters. These parameters can be understood if we explore two
concepts called Density Reachability and Density Connectivity. Reachability in terms of
density creates a point that can be reached from another point if it is located at a certain
distance (eps) from a point. Connectivity involves a transitivity based chaining-approach to
determine whether points are located in a particular cluster. For example, p and q points
could be connected if p->r->s->t->q.

  There are three types of points after the completion of the DBSCAN algorithm as core, noise
and border. Core point is the point which has at least x points within distance y from itself.
Border point is the point which has at least one core point at a distance y. Lastly, noise
point is the point which has less than x points at a distance y from itself.

How it works?
  Controls all points starting from any point(by arbitrarily) in the dataset. If the checked point
has been included in a cluster before, it moves to the other point without any action. If the
point has not been clustered before, it finds the Eps neighbors of the point by performing a
region query (Region Query). If the number of neighbors is more than MinPts, it names this
point and its neighbors as a new cluster. It then finds new neighbors by querying the new
region for each previously unclustered neighbor. If the number of neighbors of the region
query is more than MinPts, it is included in the cluster.

The complexity of DBSCAN Algorithm;

Best Case: O(nlogn)

Worst Case: O(n²)

Average Case: Same as best/worst case depending on data and implementation of the
              algorithm.
   
Note that; Instead of examining every point in the neighborhood, various indexing
algorithms such as R*- tree or spatial query have been proposed to reduce the time
complexity to O(logn).

Advantages:

  ● Resistant to noise
  
  ● Can handle cluster of different shapes and sizes
  
  ● It just needs two paremeters: MinPts ad Eps.
    Disadvantages:

  ● It does not work well varying densities and high-dimensional data
  
  ● Sensitive to paramete
  
### Balanced Iterative Reduction and Clustering Using Hierarchies (BIRCH):
  Balanced Iterative Reduction and Clustering Using Hierarchies (BIRCH) is a clustering
algorithm that can cluster large datasets by first creating a small and compact summary of the
large dataset that holds as much information as possible. This smaller summary clusters rather
than clusters the larger dataset.

  BIRCH is often used to complement other clustering algorithms by creating a summary
of the dataset that the other clustering algorithm can now use. However, BIRCH has one major
drawback - it can only handle metric attributes. A metric attribute is any attribute whose values
can be represented in Euclidean space, that is, no categorical attributes should exist.

Explanation of the important terms and parameters of the BIRCH algorithm;
  BIRCH algorithm takes three parameters as threshold, braching_factor and n_clusters.
threshold determines the maximum number of data points a sub cluster in the leaf node of the
cf tree can hold. branching_factor specifies the maximum number of CF sub clusters in each
node. n_clusters is the number of clusters to be returned after the entire BIRCH algorithm is
complete i.e., number of clusters after the final clustering step. If set to None, the final
clustering step is not performed and intermediate clusters are returned.


Clustering Feature (CF)
  BIRCH summarizes large datasets into smaller, denser regions called Clustering Feature
(CF) entries. Formally, a Clustering Feature entry is defined as an ordered triple (N, LS, SS);
where 'N' is the number of data points in the cluster, 'LS' is the linear sum of data points, and
'SS' is the squared sum of data points in the cluster. It is possible for one CF input to be
composed of other CF inputs.

CF Tree
  The CF tree is the actual compact representation that we have been speaking of so far. A
CF tree is a tree where each leaf node contains a sub-cluster. Every entry in a CF tree contains a
pointer to a child node and a CF entry made up of the sum of CF entries in the child nodes.
There is a maximum number of entries in each leaf node. This maximum number is called the
threshold.

### Spectral Clustering
  Spectral clustering is a technique based on graph theory. The approach is used to
identify communities of vertices in a graph based on the edges connecting them. This method is
flexible and allows us to cluster non-graph data as well either with or without the original data.

  The technique involves representing data in a low dimension. At lower size, the clusters
in the data are more widely separated, allowing you to use algorithms such as k-means or
k-medoids clustering. This lower dimension is based on the eigenvectors of a Laplacian matrix.
A Laplacian matrix is a way of representing a similarity graph that models local neighborhood
relationships between data points as an undirected graph. You can use spectral clustering when
you know the number of clusters, but the algorithm also provides a way to estimate the number
of clusters in your data.

## TOOLS
  ● The Natural Language Toolkit (NLTK)
  
  ● Scikit-learn
  
  ● Zemberek


