"""
================================================================================
EDU-ML SANDBOX — MODEL PLUGIN
================================================================================

@model: LogisticRegression
@task: classification
@name: Logistic Regression
@description: Despite the name "regression", this is a CLASSIFICATION model. 
              It predicts the probability of belonging to a class. It uses 
              the sigmoid (logistic) function to convert a linear result 
              into a probability of 0-100%. A great starting point for 
              binary classification — simple, interpretable, fast.
@icon: logistic_regression.svg

================================================================================
MODEL EQUATION
================================================================================

@equation: P(y=1) = 1 / (1 + e^(-(w₁x₁ + w₂x₂ + ... + b)))
@cost_function: Log Loss = -Σ[y·log(p) + (1-y)·log(1-p)]
@cost_name: Log Loss (Cross-Entropy)

================================================================================
PARAMETERS — VISIBLE TO STUDENT
================================================================================

@param: C
@label: Regularization Strength (C)
@type: float
@min: 0.01
@max: 100
@step: log
@default: 1.0
@hint: Inverse of regularization strength — SMALLER C = STRONGER regularization.
       
       SMALL C (0.01 - 0.1):
       • Strong regularization, model is "cautious"
       • Coefficients close to zero
       • Simpler decision boundary
       • Lower risk of overfitting
       • Use when you have little data or many features
       
       LARGE C (10 - 100):
       • Weak regularization, model is "confident"
       • Coefficients can be large
       • More complex boundary
       • Risk of overfitting
       • Use when you have lots of data
       
       EXPERIMENT: Start with C=1, decrease if model overfits.

@param: penalty
@label: Regularization Type
@type: select
@options: l2, l1, none
@default: l2
@hint: Method of "penalizing" the model for coefficients that are too large.
       
       L2 (Ridge):
       • Penalty proportional to the SQUARE of coefficients
       • Shrinks all coefficients, but doesn't zero them out
       • DEFAULT and recommended in most cases
       
       L1 (Lasso):
       • Penalty proportional to the ABSOLUTE VALUE
       • Can ZERO OUT some coefficients → feature selection!
       • Use when you suspect some features are unimportant
       • NOTE: Requires solver='saga'
       
       none:
       • No regularization
       • Risk of overfitting
       • Use only with large amounts of data

@param: max_iter
@label: Maximum Number of Iterations
@type: int
@min: 50
@max: 1000
@step: 50
@default: 100
@hint: How many optimization steps the model can perform.
       
       LOW VALUE (50-100):
       • Faster training
       • Risk that the model won't stabilize
       
       HIGH VALUE (500-1000):
       • Model has time to converge
       • Slower training
       
       💡 If you see a "convergence not reached" warning 
          (ConvergenceWarning), increase max_iter.
       
       EXPERIMENT: Observe n_iter_ in results — if it equals max_iter,
                   the model may not have stabilized!

@param: class_weight
@label: Class Weights
@type: select
@options: none, balanced
@default: none
@hint: How to handle imbalanced classes (e.g., 90% class A, 10% class B).
       
       none:
       • All classes equally important
       • OK when classes are balanced
       
       balanced:
       • Automatically increases weight of the rarer class
       • Model "tries harder" to correctly classify the minority
       • Use when one class is much rarer
       
       EXPERIMENT: For imbalanced data, compare results
                   with none vs balanced — see how Recall changes!

================================================================================
PARAMETERS — HIDDEN (technical)
================================================================================

@param: solver
@show: false

@param: tol
@show: false

@param: fit_intercept
@show: false

@param: intercept_scaling
@show: false

@param: random_state
@show: false

@param: dual
@show: false

@param: verbose
@show: false

@param: warm_start
@show: false

@param: n_jobs
@show: false

@param: l1_ratio
@show: false

@param: multi_class
@show: false

================================================================================
MODEL OUTPUTS — ATTRIBUTES
================================================================================

@output: coef_
@output_label: Coefficients (log-odds)
@output_type: matrix
@output_show: true
@output_format: bar_chart
@output_hint: Coefficients in log-odds scale.
              
              INTERPRETATION:
              • Coefficient of +1 means: increasing feature by 1 increases 
                log-odds by 1, so odds grow ~2.7× (e^1 ≈ 2.718)
              • Coefficient of -1 means: odds decrease ~2.7×
              • Coefficient of 0 means: feature doesn't affect the outcome
              
              SIGN:
              • Positive (+) → feature INCREASES chance of class 1
              • Negative (-) → feature DECREASES chance of class 1
              
              MAGNITUDE:
              • |coef| > 1 → strong influence
              • |coef| < 0.5 → weak influence

@output: intercept_
@output_label: Intercept (bias)
@output_type: vector
@output_show: true
@output_format: text
@output_hint: Base log-odds when all features = 0.
              
              If intercept > 0: model by default "prefers" class 1
              If intercept < 0: model by default "prefers" class 0
              
              This is the "starting point" before accounting for features.

@output: classes_
@output_label: Recognized Classes
@output_type: labels
@output_show: true
@output_format: text
@output_hint: List of classes the model distinguishes.
              Order matters — class 1 is the "positive" one 
              (for which we calculate probability).

@output: n_iter_
@output_label: Number of Iterations Performed
@output_type: integer
@output_show: true
@output_format: text
@output_hint: How many optimization iterations the model performed.
              
              ⚠️ IMPORTANT: If n_iter_ = max_iter, the model may 
              NOT have stabilized! Increase max_iter and run again.
              
              💡 If n_iter_ << max_iter, the model quickly reached 
              convergence — that's good!

@output: n_features_in_
@output_label: Number of Features
@output_type: integer
@output_show: false

@output: feature_names_in_
@output_show: false

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
              
              ✅ PROS: Intuitive, easy to understand
              ⚠️ CONS: Misleading when classes are imbalanced!
              
              Example problem: 95% class A, 5% class B
              A model always saying "A" has 95% accuracy, but is useless.

@metric: precision
@metric_label: Precision
@metric_show: true
@metric_format: percent
@metric_good_value: 0.7
@metric_hint: Of those the model marked as positive, how many actually were?
              
              Precision = TP / (TP + FP)
              
              "How much can I TRUST a positive prediction?"
              
              IMPORTANT WHEN: False alarm is costly
              • Spam filter: email marked as spam goes to trash
              • Diagnosis: false positive = unnecessary stress
              
              High Precision = few false alarms

@metric: recall
@metric_label: Recall (Sensitivity)
@metric_show: true
@metric_format: percent
@metric_good_value: 0.7
@metric_hint: Of those that were actually positive, how many did the model find?
              
              Recall = TP / (TP + FN)
              
              "How well does the model FIND positive cases?"
              
              IMPORTANT WHEN: Missing a case is costly
              • Disease detection: miss = no treatment
              • Fraud detection: miss = money loss
              
              High Recall = few misses

@metric: f1
@metric_label: F1-Score
@metric_show: true
@metric_format: percent
@metric_good_value: 0.7
@metric_hint: Harmonic mean of Precision and Recall.
              
              F1 = 2 × (Precision × Recall) / (Precision + Recall)
              
              WHY HARMONIC?
              • Penalizes extreme values
              • F1=0.9 requires BOTH metrics to be high
              • If one = 0, F1 = 0
              
              USE WHEN: You care about balance between 
              false alarms and misses.

@metric: roc_auc
@metric_label: ROC AUC
@metric_show: true
@metric_format: decimal
@metric_good_value: 0.8
@metric_hint: Area Under the ROC Curve (Receiver Operating Characteristic).
              
              INTERPRETATION:
              • 0.5 = random guessing (useless model)
              • 0.7-0.8 = acceptable model
              • 0.8-0.9 = good model
              • 0.9+ = excellent model
              • 1.0 = perfect separation
              
              ADVANTAGE: Doesn't depend on decision threshold!
              Measures how well the model RANKS examples
              (whether positives have higher probabilities than negatives).

@metric: log_loss
@metric_label: Log Loss
@metric_show: false
@metric_format: decimal
@metric_hint: Cost function used during training.
              The smaller, the better the model predicts probabilities.

================================================================================
VISUALIZATIONS
================================================================================

@visualization: equation
@viz_label: Logit Equation
@viz_show: true
@viz_position: top

@visualization: decision_boundary
@viz_label: Decision Boundary
@viz_show: true
@viz_position: main

@visualization: coefficients_bar
@viz_label: Feature Impact (bar chart)
@viz_show: true
@viz_position: side

@visualization: coefficients_table
@viz_label: Coefficients Table
@viz_show: true
@viz_position: side

@visualization: confusion_matrix
@viz_label: Confusion Matrix
@viz_show: true
@viz_position: side

@visualization: roc_curve
@viz_label: ROC Curve
@viz_show: true
@viz_position: bottom

@visualization: probability_distribution
@viz_label: Probability Distribution
@viz_show: true
@viz_position: bottom

@visualization: precision_recall_curve
@viz_label: Precision-Recall Curve
@viz_show: true
@viz_position: bottom

================================================================================
EDUCATIONAL INFORMATION
================================================================================

LEARNING OBJECTIVE:
Understanding probabilistic classification and the logistic function.

KEY CONCEPTS:
• Sigmoid function: σ(z) = 1/(1+e^(-z)) — converts any number to 0-1
• Log-odds (logit): log(p/(1-p)) — logarithm of odds
• Probability vs. prediction: model gives P, threshold converts to class
• Regularization: penalizing large coefficients

DIFFERENCE FROM LINEAR REGRESSION:
• Linear regression: predicts a NUMBER (price, age)
• Logistic regression: predicts a PROBABILITY (0-100%)

EXPERIMENT QUESTIONS:
1. How does the decision boundary change when you decrease C?
2. What happens when you enable L1 regularization? Which coefficients disappear?
3. How does class_weight="balanced" affect Recall of the minority class?
4. When does n_iter_ = max_iter? What does it mean?

================================================================================
"""

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
