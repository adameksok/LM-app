'''
================================================================================
EDU-ML SANDBOX — MODEL PLUGIN
================================================================================

@model: KNeighborsClassifier
@task: classification
@name: K-Nearest Neighbors (KNN)
@description: Classifies new data points based on "voting" by the K nearest 
              points from the training set. A lazy learning algorithm — it does not build
              an equation during training, but only memorizes the dataset for comparisons.
@icon: knn.svg

================================================================================
MODEL EQUATION
================================================================================

@equation: r"y = M_{k}(\mathbf{x})"
@cost_function: No classical cost function optimization
@cost_name: Lazy Learning

================================================================================
PARAMETERS — VISIBLE TO STUDENT
================================================================================

@param: n_neighbors
@label: Number of Neighbors (K)
@type: int
@min: 1
@max: 51
@step: 2
@default: 5
@hint: How many nearest points participate in the voting.
       
       SMALL K (e.g., 1-3):
       • Decision boundary is very "jagged" and fits to every point
       • Very high risk of overfitting
       • Captures even isolated noise
       
       LARGE K (e.g., 15-25):
       • Smooth, generalized decision boundary
       • Resistant to noise (smooths out individual outliers)
       • Too large K can lead to underfitting
       
       EXPERIMENT: Set K=1 and observe islands around individual points.

@param: weights
@label: Neighbor Weights
@type: select
@options: uniform, distance
@default: uniform
@hint: How to count neighbor votes.
       
       UNIFORM (equal):
       • Each of the K neighbors has exactly the same 1 vote
       • Standard, simple behavior
       
       DISTANCE (weighted by distance):
       • The closer the point, the stronger its vote (weighs more)
       • Very useful when the dataset is imbalanced and small classes are clustered, 
         while dominant classes "flood" the surroundings

@param: p
@label: Distance Metric
@type: select
@options: Manhattan:1, Euclidean:2
@default: 2
@hint: How the space "from one point to another" is measured.
       
       1 (Manhattan / City Block):
       • Calculates distance like steps on a grid of streets (|x1-x2| + |y1-y2|)
       • Sometimes better for rigidly defined feature grids
       
       2 (Euclidean / Straight Line):
       • Classic straight-line distance from point to point
       • DEFAULT CHOICE IN 99% OF CASES

================================================================================
PARAMETERS — HIDDEN (technical)
================================================================================

@param: algorithm
@show: false

@param: leaf_size
@show: false

@param: metric
@show: false

@param: metric_params
@show: false

@param: n_jobs
@show: false

================================================================================
MODEL OUTPUTS — ATTRIBUTES
================================================================================

@output: classes_
@output_label: Recognized Classes
@output_type: labels
@output_show: true
@output_format: text
@output_hint: List of classes the model distinguishes and subjects to "voting".

@output: n_samples_fit_
@output_label: Memorized Base Points
@output_type: integer
@output_show: true
@output_format: text
@output_hint: KNN does not use coefficients or intercepts.
              KNN requires memorizing the ENTIRE training dataset 
              to operate (so-called lazy learning).
              The model loaded the shown number of reference points into memory.

@output: effective_metric_
@output_label: Applied Mathematical Distance Metric
@output_type: string
@output_show: true
@output_format: text
@output_hint: The computational metric embedded in the algorithm's backend.

================================================================================
QUALITY METRICS
================================================================================

@metric: accuracy
@metric_label: Accuracy
@metric_show: true
@metric_format: percent
@metric_good_value: 0.8
@metric_hint: Percentage of correctly classified samples.
              Accuracy = (TP + TN) / (TP + TN + FP + FN)

@metric: precision
@metric_label: Precision
@metric_show: true
@metric_format: percent
@metric_good_value: 0.7
@metric_hint: Of those the model marked as positive, how many actually were?
              Precision = TP / (TP + FP)

@metric: recall
@metric_label: Recall (Sensitivity)
@metric_show: true
@metric_format: percent
@metric_good_value: 0.7
@metric_hint: Of those that were actually positive, how many did the model find?
              Recall = TP / (TP + FN)

@metric: f1
@metric_label: F1-Score
@metric_show: true
@metric_format: percent
@metric_good_value: 0.7
@metric_hint: Harmonic mean of Precision and Recall.
              F1 = 2 × (Precision × Recall) / (Precision + Recall)

@metric: roc_auc
@metric_label: ROC AUC
@metric_show: true
@metric_format: decimal
@metric_good_value: 0.8
@metric_hint: Area Under the ROC Curve (Receiver Operating Characteristic).
              0.5 = coin flip, 1.0 = perfect prediction.

================================================================================
VISUALIZATIONS
================================================================================

@visualization: decision_boundary
@viz_label: Neighborhood Decision Boundaries
@viz_show: true
@viz_position: main

@visualization: confusion_matrix
@viz_label: Confusion Matrix
@viz_show: true
@viz_position: side

@visualization: roc_curve
@viz_label: ROC Curve
@viz_show: true
@viz_position: bottom

@visualization: probability_distribution
@viz_label: Voting Probability Distribution
@viz_show: true
@viz_position: bottom

@visualization: precision_recall_curve
@viz_label: Precision-Recall Curve
@viz_show: true
@viz_position: bottom

================================================================================
'''

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
