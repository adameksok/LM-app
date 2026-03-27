"""
================================================================================
EDU-ML SANDBOX — MODEL PLUGIN
================================================================================

@model: LinearRegression
@task: regression
@name: Linear Regression
@description: The simplest and most intuitive regression model. It fits a 
              straight line to the data using the Ordinary Least Squares (OLS) 
              method. Ideal for understanding the basics of machine learning — 
              see how the line "follows" the points and what happens when 
              you add an outlier.
@icon: linear_regression.svg

================================================================================
PARAMETERS
================================================================================

@param: fit_intercept
@label: Fit intercept
@type: bool
@default: true
@hint: Determines whether the line can cross the Y-axis at any point.
       
       ENABLED (True):
       • Line form: y = a·x + b
       • The "b" parameter (intercept) is fitted
       • Line can start above or below zero
       • RECOMMENDED for most cases
       
       DISABLED (False):
       • Line form: y = a·x
       • Line MUST pass through the point (0, 0)
       • Use only when you know the data starts from zero
       
       EXPERIMENT: Disable it and see how the line "jumps" to the point (0,0).

@param: positive
@label: Positive coefficients only
@type: bool
@default: false
@hint: Forces the coefficients (slopes) to be non-negative.
       
       DISABLED (False):
       • Line can have any slope (increasing or decreasing)
       • RECOMMENDED for most cases
       
       ENABLED (True):
       • Line can only be increasing or horizontal
       • Use when you know the relationship must be positive
         (e.g., more study hours = higher score)
       
       EXPERIMENT: Enable for data with a downward trend and observe the effect.

================================================================================
HIDDEN PARAMETERS (technical, irrelevant for learning)
================================================================================

@param: copy_X
@show: false

@param: n_jobs
@show: false

@param: tol
@show: false

================================================================================
EDUCATIONAL INFORMATION
================================================================================

LEARNING OBJECTIVE:
Understanding how a model "learns" from data by minimizing error.

KEY CONCEPTS:
• Slope — how much y increases when x increases by 1
• Intercept — where the line crosses the Y-axis
• Mean Squared Error (MSE) — a measure of fit quality
• R² — the percentage of variance explained by the model

EXPERIMENT QUESTIONS:
1. What happens when you add a point very far from the rest (an outlier)?
2. How does the slope of the line change when you disable the intercept?
3. Why does MSE increase when the line does not fit the data?

================================================================================
"""

from sklearn.linear_model import LinearRegression

model = LinearRegression()