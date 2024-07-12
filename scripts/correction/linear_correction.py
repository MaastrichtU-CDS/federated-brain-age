# Linear correction for brain age
# Based on the approach proposed by following paper:
# https://doi.org/10.1016/j.neuroimage.2020.117292

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ba_train = json.load(open('./file1.json'))
ba_train_gap = json.load(open('./file2.json'))
ba_test = json.load(open('./file3.json'))
ba_test_gap = json.load(open('./file4.json'))

def calculate_age(brain_age, age_gap):
    for id in brain_age.keys():
        ages_train.append(brain_age[id] - age_gap[id])

def apply_linear_correction(reg, brain_age, ages):
    predicted_age_corrected = []
    age_gap = []
    age_gap_abs = []
    correction = reg.predict([[age] for age in ages])
    ids = list(brain_age.keys())
    for i in range(len(brain_age.keys())):
        id = ids[i]
        age_corrected = brain_age[id] - correction[i]
        predicted_age_corrected.append(age_corrected)
        age_gap.append(ages[i] - age_corrected)
        age_gap_abs.append(abs(ages[i] - age_corrected))
    return age_corrected, age_gap, age_gap_abs


ages_train = calculate_age(ba_train, ba_train_gap)
gap = ba_train_gap.values()

# Fit the linear regression
reg = LinearRegression().fit([[age] for age in ages_train], gap)
print(reg.coef_)
print(reg.intercept_)
# Test the prediction
pred_age = reg.predict([[40], [45], [50], [55], [60], [65], [70], [75]])
print(pred_age)

# Apply it to the dataset - Training set
predicted_age, age_gap, age_gap_abs = apply_linear_correction(reg, ba_train, ages_train)
print(np.average(age_gap))

# Apply it to the dataset - Testing set
ages_test = calculate_age(ba_test, ba_test_gap)
predicted_age, age_gap, age_gap_abs = apply_linear_correction(reg, ba_test, ages_test)
print(np.average(age_gap))

# Plot
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle

data = {'age': ages_test, 'brain age': predicted_age}
df = pd.DataFrame(data)
df.sort_values('age', ascending=True, inplace=True)

fig, ax = plt.subplots()
plt.scatter("age", "predicted_age", data=df,s=1)
plt.xlabel('Chronological Age', fontweight='bold')
plt.ylabel('Brain Age', fontweight='bold')
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlim(35, 80)
plt.ylim(35, 80)
plt.show()
