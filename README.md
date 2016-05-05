# Completed Projects for Udacity ML Engineer Nanodegree
This repository consists of five projects completed for the Udacity Machine Learning Engineer Nanodegree. A short outline of the projects is provided below.

## 1. Boston Housing Prices
The goal of this project is to accurately predict house prices in the Boston area. A decision tree regressor is trained to achieve this task.

## 2. Building a Student Intervention System (Supervised Learning)
Goal of this project is to identify an efficient machine learning algorithm to identify students that may benefit from early intervention in order to pass a test. Algorithms and models such as logistic regression, support vector classifier and random forrest classifiers are applied. Given the tradeoff between efficiency and accuracy, logistic regression is recommended for this problem.

## 3. Creating Customer Segments (Unsupervised Learning)
Goal of this project is to analyze and segment customers of a wholesale distributor. Various techniques, such as PCA, ICA and k-means clustering are applied to accurately cluster customers.

## 4. Training a Smartcab how to drive (Reinforcement Learning)
Goal of this project is to devise and algorithm that allows an agent that moves on a grid with obstacles (Smartcab) to learn how to reach a destination on the grid, avoiding obstacles on the way. Q-learning is applied to implement agent learning and learning parameters are tuned such that the agent learns very quickly.

## 5. Capstone Project: Predicting Irish property transaction prices
**Project Overview**
The goal of this project is to analyze and predict house prices in Ireland and to find a machine learning algorithm that has good predictive performance for Irish property prices. The main dataset used is the Irish residential property register. It includes Date of Sale, Price, Address and some other features of all residential properties purchased in Ireland since the 1st January 2010, as declared to the Revenue Commissioners for stamp duty purposes. A subset of this dataset (2010 until the end of 2012) is preprocessed, geocoded, linked to police stations and local economic indicators and enriched with local crime statistics from the Irish Central Statistics office for each police station. Thereafter, various machine learning models are trained and refined to find a model that accurately predicts the transaction price of a property from its features.

**Problem Statement and Approach**
Find a machine learning algorithm (and corresponding hyperparameters) that accurately predicts Irish property transaction prices from the given (and some constructed) features. The first step to achieve this is to initialize and preprocess the data and to map additional features that may have predictive power, such as local income per capita and crime statistics (Section 1). The next step is to explore and visualize the main features in the data and their relationships (Section 2). The third step is to run various regression algorithms in order to get a big picture of their performance on this dataset (Section 3). The fourth step is to perform hyperparameter tuning for the parameters of the best performing algorithms in the previous step (Section 4). Finally, in the last step, the results of analyzing predictive performance for the tuned algorithms applied to a test data set are reported (Section 5).
The expectation is that some of the constructed features, such as income per capita, have strong predictive power. Moreover, it is expected that a random forest algorithm is among the best performing ones, since these types of algorithms are commonly used for predicting house prices.

**Main Result**
We find that an optimized random forest algorithm performs best on the test set with an MSE of 0.369 and thus is selected as the most accurate predictor (in a mean-squared-sense) for log-house-prices in Ireland based on our analysis. This result confirms the expectation that random forest algorithms generally perform well for property price predictions. 