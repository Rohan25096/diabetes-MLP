from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
fpr,tpr,thresholds = roc_curve(y_test,y_log)

import plotly.graph_objects as go
import numpy as np

# Assuming fpr, tpr, and thresholds are defined here
# For example:
# fpr, tpr, thresholds = some_function_to_compute_roc()

# Generate a trace for ROC curve
trace0 = go.Scatter(
    x=fpr,
    y=tpr,
    mode='lines',
    name='ROC curve'
)

# Only label every nth point to avoid cluttering
n = 19
indices = np.arange(len(thresholds)) % n == 0

trace1 = go.Scatter(
    x=fpr[indices],
    y=tpr[indices],
    mode='markers+text',
    name='Threshold points',
    text=[f"Thr: {thr:.2f}" for thr in thresholds[indices]],
    textposition='top center'
)

# Diagonal line
trace2 = go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Random (Area = 0.5)'
)


fig = go.Figure()
fig.add_trace(trace0)
fig.add_trace(trace1)
fig.add_trace(trace2)


fig.show()

#Setting the optimal threshold.
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Optimal Thresholds is: ",optimal_threshold)

#Evaluating the test accuracy.
y_pred = np.where(y_log>optimal_threshold,1,0)
acc = accuracy_score(y_test,y_pred)
print(f"Accuracy: {acc*100:.2f}%")

#Predicting the loss vs epoch graph.
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Model loss')
plt.legend()
plt.show()

#Predicting the accuracy vs epoch graph.
plt.plot(history.history['accuracy'],label='training_acc')
plt.plot(history.history['val_accuracy'],label='val_acc')
plt.legend()
plt.show()
