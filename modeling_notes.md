# Multi-class and multi-label classification
Also see:
* http://gen.lib.rus.ec/book/index.php?md5=7D9A914B11899C853D5CF1DB8361CB83
* http://gen.lib.rus.ec/book/index.php?md5=9880D6720C107975994348750F4E2495
* https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/
* http://scikit-learn.org/stable/modules/multiclass.html

## Types
Multi-class classification:
* There are multiple (i.e. more than two) mutually exclusive categories
* Each example is assigned only to one category

Multi-label classification:
* Categories are non-exclusive
* An example can be assigned to multiple categories

Multi-class multi-output: The combination of multi-label and multi-class:
* There are multiple categories (i.e. more than two)
* One example can have more than one category

## Solution strategies
### Transformation to binary
Transforming multiclass into multiple binary classification problem
* One vs One:
    - training a single classifier fore each class
    - this needs confidence scores
    - metric: micro average (ratio of the sum of true positives with the sum of positive predictions)
* One vs All: one trains K (K − 1) / 2 binary classifiers for a K-way multiclass problem
    - metric: macro average (average prediction accuracy)

### Extension from binary
* Neural networks
* Extreme learning machines
* K-nearest neighbors
* Naive bayes
* Decision trees
* Support Vector Machines

### Hierchical classifier

## Learning algoriths
Batch learning: Trains the model on the whole dataset.

Online learning: Incrementally build the models in iterations.
* related technique: "progressive learning technique"

# Decision Tree
## Multi-output problem
* Store n output values in leaves, instead of 1
* Use splitting criteria that compute the average reduction across all n outputs

## Tips for practical use
Danger of overfitting:
* right ratio of samples/features
* use dimensionality reduction (e.g. PCA)
* visualize with `export`

* constrain the depth of the tree by `max_depth`
* Use `min_samples_split` or `min_samples_leaf` to control the number of samples at a leaf node.
* Balance the dataset
    - Sample an equal number of samples from each class
    - Normalize sample weights for each class with `sample_weight`.

# Performance metrics
