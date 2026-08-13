# Instagram Reels Performance Prediction Model

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)

An end-to-end Machine Learning pipeline developed in Python to predict Instagram Reels engagement rates. By evaluating content metadata, metrics, and temporal factors, this model provides creators and marketers with data-driven insights to optimize content strategy and maximize reach.

---

## 📌 Project Overview

With short-form video content dominating social media landscapes, understanding the mechanics behind virality is essential. This repository contains a multiple linear regression model that analyzes structural and contextual features of Instagram Reels to accurately forecast user engagement. 

### Key Predictors (Features):
*   `video_length_sec`: Total duration of the reel in seconds.
*   `editing_quality_score`: A normalized scale evaluation of video production quality.
*   `posting_hour`: 24-hour timestamp of when the content was published.
*   `hashtag_count`: Total number of embedded hashtags in the caption.
*   `follower_count`: The total following size of the publishing account.
*   `caption_length`: Character length of the textual caption accompaniment.

### Target Variable:
*   `engagement_rate`: The normalized target metric representing likes, comments, shares, and saves relative to profile reach.

---

##  Performance Metrics

The model achieves strong tracking performance between contextual metadata and actual audience reception:

*   **R² Score (Hold-out Validation):** ~`0.707` (The features account for ~71% of the total variance in engagement rate).
*   **R² Score (Entire Dataset):** ~`0.684`

---

##  Tech Stack & Dependencies

*   **Core Language:** Python
*   **Data Processing:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn
*   **Data Visualization:** Matplotlib

---

##  Getting Started

### 1. Prerequisites & Installation

Ensure you have Python installed, then clone the repository and install the verified environment requirements:

```bash
# Clone the repository
git clone https://github.com/yourusername/ig-reel-performance-model.git
cd ig-reel-performance-model

# Install necessary libraries
pip install numpy pandas matplotlib scikit-learn
```

### 2. Dataset Setup
Place your dataset (`instagram_reels_dataset.csv`) inside the project structure. The data pipeline expects a clean tabular configuration containing features and the target variable as outlined in the overview.

---

##  Code Architecture & Execution

The workflow is written in an exploratory/production Jupyter Notebook environment. Below is the structured breakdown of the pipeline's core logic:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection

# 1. Data Ingestion & Inspection
df = pd.read_csv('instagram_reels_dataset.csv')
print(df.info())
print(df.isnull().sum())

# 2. Train-Test Split Matrix
X = df[['video_length_sec', 'editing_quality_score', 'posting_hour', 'hashtag_count', 'follower_count', 'caption_length']]
y = df['engagement_rate']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=45)

# 3. Model Initialization & Training
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)

# 4. Evaluation Evaluation 
print(f"Validation R² Score: {lr.score(X_test, y_test):.3f}")

# 5. Production Prediction Inference (Example case)
# Input shape template: [[video_length, editing_score, hour, hashtags, followers, caption_len]]
sample_reel = [[30, 8, 12, 4, 1000, 25]]
predicted_er = lr.predict(sample_reel)
print(f"Predicted Engagement Rate: {predicted_er[0]:.2f}%")
```

---

# Author
Waqas ali khan

