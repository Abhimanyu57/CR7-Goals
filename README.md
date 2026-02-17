
Classifying Match Outcomes Using Cristiano Ronaldo's Club Goal Data
Project Overview

This project explores whether historical goal-level data from Cristiano Ronaldo’s club career contains enough contextual information to classify final match outcomes (Win, Draw, Loss) using supervised learning techniques.

The goal is not real-world match prediction. Instead, the project focuses on:

Demonstrating structured data preprocessing and feature handling

Comparing fundamentally different machine learning models

Understanding how model selection influences interpretability, bias–variance tradeoffs, and performance

This is an exploratory and educational study designed to show how structured sports event data can be transformed into a meaningful classification problem.

Objective

The central question is:

Can contextual signals surrounding individual goal events help classify the final result of a match?

By reframing goal-level football data as a classification task, the project evaluates how effectively machine learning models can extract predictive structure from limited contextual features.

Dataset Description

The dataset consists of goal-level records from Cristiano Ronaldo’s club career. Each row represents a single goal scored.

Available features include:

Season

Competition

Matchday

Venue

Opponent

Minute of the goal

Score context at the time of scoring

Final match result

Since the dataset is goal-centric rather than match-centric, a single match may appear multiple times. The task therefore evaluates whether partial contextual information around goals correlates with the final outcome of the match.

Data Preparation

To reduce noise and overly specific signals, certain features such as playing position and goal assist were removed.

Key preprocessing steps include:

Converting match results into a categorical target variable (Win, Draw, Loss)

One-hot encoding categorical features

Passing numerical features through without scaling

Using scikit-learn pipelines to ensure proper preprocessing and prevent data leakage

All transformations are encapsulated within model pipelines to maintain reproducibility and structural clarity.

Models Implemented

Two fundamentally different models were selected for comparison:

1. Logistic Regression

A linear model used as a strong and interpretable baseline.
It provides stable decision boundaries and serves as a reference point for performance evaluation.

2. Decision Tree Classifier

A non-linear, rule-based model capable of capturing feature interactions and complex decision patterns.
It also allows direct inspection of learned decision rules.

The comparison highlights tradeoffs between simplicity, interpretability, flexibility, and variance.

Key Insights

Contextual goal-level data contains meaningful signal for classifying match outcomes.

Logistic Regression performs reliably as a stable baseline.

The Decision Tree captures non-linear patterns but exhibits higher variance.

Performance differences across outcome classes reveal class imbalance, particularly for draws.

These findings demonstrate both the potential and limitations of using partial football event data for classification tasks.

Limitations

This analysis does not incorporate full match context such as:

Minutes played

Team strength

Opponent quality

Tactical setup

Because the dataset is goal-level rather than match-level, it cannot fully characterize match dynamics.

Future Work

Potential extensions include:

Aggregating data at the match level

Incorporating richer contextual features

Addressing class imbalance explicitly

Comparing ensemble methods such as Random Forests or Gradient Boosting

These improvements could enhance both predictive performance and analytical depth.
