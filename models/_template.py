"""
================================================================================
EDU-ML SANDBOX — PLUGIN TEMPLATE
================================================================================

Copy this file, rename it, and customize the @tags below.
Minimum required: @model and @task.
Save into /models folder — it will appear on the dashboard automatically.

@model: KNeighborsClassifier
@task: classification
@name: Template Model
@description: This is a template. Replace with your own model.

@param: n_neighbors
@label: Number of Neighbors (K)
@type: int
@min: 1
@max: 30
@step: 1
@default: 5
@hint: Higher K = smoother boundary, lower K = more complex boundary.
"""
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
