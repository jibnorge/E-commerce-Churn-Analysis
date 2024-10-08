# E-commerce-Churn-Analysis
This repository contains a comprehensive churn analysis for an e-commerce website. The analysis focuses on identifying patterns and reasons behind customer churn, applying machine learning models, and providing actionable insights to reduce churn.  Key areas include EDA, feature engineering, model building, and evaluation metrics.

## Table of Contents
1. [Introduction](#introduction)
2. [Need of Study](#need-of-study)
3. [Dataset](#dataset)
4. [Code Usage](#code)
5. [Tools & Techniques](#tools-techniques)
6. [Data Preperation and Understanding](data-prep)
    - [Phase I - Data Extraction and Cleaning](phase-1)
    - [Phase II - Exploratory Data Analysis](#phase-2)
    - [Phase III - Feature Engineering](#phase-3)
7. [Fitting Models to the Data](model-fitting)
    - [Linear Regression](#lin-reg)
    - [Decision Tree](#dt)
    - [Random Forest](#rf)
    - [KNN](#knn)
    - [Ada Boost](#ada-boost)
    - [Gradient Boost](#gradient-boost)
    - [Light GBM](#light-gbm)
    - [Cat Boost](#cat-boost)
    - [XG Boost](#xg-boost)
8. [Key Findings](#key-findings)
9. [Recommendations](#recommendation)
10. [Conclusion](#conclusion)

<a name="introduction"></a>
## Introduction 
The objective of the E-commerce Churn Analysis project is to develop a machine learning model that accurately predicts customer churn on an e-commerce platform. Churn refers to the likelihood of a customer discontinuing their engagement or purchases from the platform. The model leverages various features such as customer demographics, purchasing behavior and interaction history to predict churn risk. The goal is to provide actionable insights to help the business retain valuable customers by identifying patterns leading to churn and implementing targeted retention strategies.

<a name="need-of-study"></a>
## Need of Study
The study is needed to address the growing challenge of customer retention in the e-commerce industry, where high competition and shifting customer preferences make it increasingly difficult to maintain customer loyalty. With the rise of online shopping options, e-commerce businesses face the risk of customer churn, which can significantly impact profitability. By developing a machine learning model that accurately predicts customer churn based on various features such as purchasing behavior, engagement patterns, and demographic data, this study aims to provide valuable insights into customer attrition.

<a name="dataset"></a>
## Dataset
The dataset used for this E-commerce Churn Analysis project is sourced from Kaggle and is made available under the CC BY-NC-SA 4.0 license. It contains customer information, purchasing behaviors, and engagement data, which are used to predict customer churn and provide actionable insights for retention strategies.

### Attribution
Dataset provided by ankitverma2010 on Kaggle.

Copyright © 2020. Licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>.

Link to the dataset: https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction/data.

Disclaimer: The dataset is provided “as-is” without any warranties or representations.

### Citation
If you use this dataset in your work, please consider acknowledging Kaggle and the dataset providers. A suggested citation format is:

Dataset Provider. (Year). "Name of the Dataset," Kaggle. URL: https://www.kaggle.com/

Please ensure that you comply with the terms of the CC BY-NC-SA 4.0 license, which allows for sharing and adapting the dataset for non-commercial purposes with appropriate attribution.

<a name="code"></a>
## Code Usage 

1. Setting up the environment
- Clone the repository 
```bash
git clone https://github.com/jibnorge/E-commerce-Churn-Analysis.git
cd E-commerce-Churn-Analysis
```

- Create a virtual python environment
```bash
python -m venv .venv
```

- Activate the environment and install requirements.txt
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

- Open jupyter notebook and run the **churn.ipynb** file.


<a name="tools-techniques"></a>
## Tools & Techniques

### Tools
- Python
- Pandas 
- NumPy
- Seaborn
- scikit-learn
- Plotly
- Folium
- SciPy

### Techniques
To evaluate the performance of classification models we use classification report and confusion metrics as evaluation metrics.

<a name="data-prep"></a>
## Data Preperation and Understanding
One of the first steps engaged in was to outline the sequence of steps that will be following for the project. Each of these steps are elaborated below:

<a name="phase-1"></a>
### Phase I - Data Extraction and Cleaning
- Reading the dataset using Pandas
- Identifying and handling missing values and duplicates
- Checking for data inconsistencies and correcting them
- Converting data types as necessary
- Dropping irrelevant or redundant columns

<br><br>

<img src="images\summary 1.png" alt="descriptive-statistics"></img>

The shape of the dataset is : (48895, 16)

The descriptive statistics show that there are outliers in most of the numerical columns. After checking for missing values, it was found that there were some in the columns named "last review" and "reviews per month.

The "reviews per month" is a numerical column with a minimum value of 0.01 and a maximum value of 58.5, with the 75th percentile as 2.02. Due to the high level of variability in the column and the presence of outliers, using simple imputation techniques like mean, median, or mode is not viable. Since the datapoints are missing completely at random, KNN Imputation techniques are used to impute values to this column.

The "last review" is a date column indicating the last date on which the review was posted for the particular listing. Since it is a date column and has no significant value to the price [target], it is better to split the values into month and year separately. The year column contains only data from 2011 to 2019, and there are 12 months in a year. As was done for the above column, in this case also the KNN 
imputation technique was used to fill in missing values for both columns since in this case also the data points were completely missing at random.

The first four columns (id, name, host name, and host id) in the dataset are excluded from the model-building process as they are considered to provide descriptive or identifying information rather than useful attributes for predicting listing prices.

The treatment for outliers is done in the coming sections, where box-cox and power transformations are used to convert the columns into a normal distribution.


<a name="phase-2"></a>
### Phase II - Exploratory Data Analysis
- Performing univariate, bivariate and multivariate analysis to understand the data
- Creating visualizations to summarize and present the data
- Calculating summary statistics such as mean, median and standard deviation to describe the data

<br><br>

The longitude and latitude columns, along with the neighbourhood group containing 5 areas in NYC (Manhattan, Queens, Brooklyn, Staten Island, and Bronx) and price, are used to create a map using the package folium to understand the areas in NYC where the Airbnb listings are located. If we zoom into these values and hover over any points, we can see the respective neighbourhood group along with its price

<img src="images\map final.png" alt="map-of-new-york-city-with listings marked"></img>

The two main categorical columns that can be seen in the data are neighbourhood group and room type. The frequency plots of these two attributes are as follows :

<img src="images\neigh_group and room.png" alt="neighbourhood groups and room type"></img>

The percentage of room type containing 3 unique values [Private room, Entire home/apt and Shared room] for each neighbourhood can be demonstrated using a pie chart.

<img src="images\room type in each neigh group.png" alt="distribution of room type in each neighbourhood group"></img>

The pie chart shows that almost all the neighbourhood groups contain above 50% of private rooms, except Manhattan, where the proportion of homes and apartments is higher.

Next comes the distribution of the target variable, which is the rental price, in each neighbourhood group and for each room type.

<img src="images\distribution of price.png" alt="distribution of price"></img>

These two violin plots clearly indicates the presence of outliers in the price column.

For numerical columns boxplots and histograms can be used to understand the distribution of each attribute. An example of these two plots for the attribute availability 365 is shown below:

<img src="images\availability 365.png" alt="distribution of avalability of listings"></img>

The final observations from ploting these two graphs for all the numerical columns are as follows:

- All the variables except availability_365 is extremely right skewed
- Except availability_365 all other numerical features have outliers.

The presence of multicolinearity can lead to unreliable, unstable, and inaccurate regression models, which can hinder our ability to make accurate predictions and draw meaningful conclusions from our data. A simple heatmap for the pairwise correlation of each attribute can be used to understand the relationship between them.

<img src="images\corr.png" alt="correlation-map"></img>

All the values in the heatmap are either less than 0.2 or greater than -0.2, which is almost close to zero. This shows that all the attributes are weakly correlated with  all other attributes.

A scatter plot can be used to analyse the relationship between two numerical columns. Taking price in the y-axis and rest of each numerical column in the x-axis along with different colour for different categories in room type scatter plots can be drawn as follows:

<img src="images\price vs num in room type.png" alt="price vs features in room type"></img>

The problem that arises here is that there is no linear relationship that can be found for any of the numerical columns with respect to the target variable. This problem can be solved by using different transformation methods on the target variable.

<a name="phase-3"></a> 
### Phase III - Feature Engineering

The number of outliers in the target variable is checked using a box plot and is 2977 of 48895 records. So, it is better to remove these records from the original data for better predictions.

Since all the numerical columns are extremely right skewed with a lot of outliers, box-cox and boxcox1p using power transformer are used to transform the data points into normal curves.

The box-cox transformation is used for attributes that are strictly positive; that is, zeros also cannot be included. The attributes “minimum nights” and “calculated host listings count” are transformed using simple box-cox method.

In situations where the data points contain zero or negative values, boxcox1p along with a power transformer can be used to convert the data into normal curves. First, the power transformer is fitted into the data points to find out the lambda values, which are then used in boxcox1p to transform the respective columns into normal curves. All other numerical columns except the two mentioned above contain zeros and are thus transformed using this method.

A small example of transformation for the column price, before and after is given below:

<img src="images\boxcox-before-and-after.png" alt="price-transformation"></img>

When dealing with categorical columns in nominal scale, such as "neighbourhood" and "neighbourhood group", the method of label encoding was applied. On the other hand, for columns in ordinal scale, such as "room type", one-hot encoding was implemented.

<a name="model-fitting"></a>
## Fitting Models to the Data

The train-test split method was used to evaluate the performance of machine learning models. This method involves splitting the available dataset into two 
parts: a training set and a testing set. The training set,which accounted for 70% of the data, was used to train the machine learning models, while the remaining 30% was used for testing the models. The training set had 32,142 records, and the testing set had 13,776 records. The train-test split allowed for the evaluation of the machine learning models on new, unseen data, which is essential for determining their effectiveness and generalizability.

For a quick head start, an ExtraTreesRegressor was built on the data to understand the features that are important for model building. The result of this model when plotted onto a bar graph was as follows:

<img src="images\newplot.png" alt="extra-tree-regressor"></img>

The feature private room has a higher contribution for predicting price followed by longitude, latitude, etc...

For each of the models given below, a GridSearchCV or RandomizedSearchCV was used to find the best parameters suitable for the models.

<a name="lin-reg"></a>
### Linear Regression
A simple linear model that attempts to predict the relationship between a dependent variable and one or more independent variables through a linear equation.

Best Parameters: {'fit_intercept': True}

MAE : 0.54; MSE : 0.50; RMSE : 0.71; R2 : 0.52

<a name="dt"></a>
### Decision Tree
A tree-structured model that breaks down a dataset into smaller and smaller subsets based on a set of decisions or rules until the subsets contain instances with a single class or value.

Best Parameters: {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2}

MAE : 0.52; MSE : 0.45; RMSE : 0.67; R2 : 0.56

<a name="rf"></a>
### Random Forest
An ensemble model that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

Best Parameters: {'n_estimators': 130, 'min_samples_split': 9, 'min_samples_leaf': 6, 'max_features': 10, 'max_depth': 10, 'bootstrap': True}

MAE : 0.47; MSE : 0.39; RMSE : 0.63; R2 : 0.62

<a name="knn"></a>
### KNN
A non-parametric model that predicts the value of a data point based on the values of its nearest neighbors in the training data.

Best Parameters: {'weights': 'distance', 'p': 1, 'n_neighbors': 13, 'leaf_size': 44, 'algorithm': 'brute'}

MAE : 0.60; MSE : 0.60; RMSE : 0.77; R2 : 0.43

<a name="ada-boost"></a>
### Ada Boost
A boosting algorithm that combines multiple weak learners into a strong learner through weighted voting to improve prediction accuracy.

Best Parameters: {'n_estimators': 400, 'learning_rate': 0.013848863713938732, 'base_estimator': DecisionTreeRegressor(max_depth=2)}

MAE : 0.58; MSE : 0.55; RMSE : 0.74; R2 : 0.47

<a name="gradient-boost"></a>
### Gradient Boost
A boosting algorithm that combines multiple weak learners to make a strong learner through an additive model, where each new learner corrects the errors of the previous one.

Best Parameters: {'subsample': 0.8999999999999999, 'n_estimators': 600, 'min_samples_split': 6, 'max_depth': 7, 'learning_rate': 0.018307382802953697}

MAE : 0.46; MSE : 0.38; RMSE : 0.62; R2 : 0.63

<a name="light-gbm"></a>
### Light GBM
A gradient boosting framework that uses a tree-based learning algorithm and aims to improve efficiency, accuracy, and speed by using a novel technique called 
Gradient-based One-Side Sampling (GOSS).

Best Parameters: {'num_leaves': 38, 'n_estimators': 170, 'min_data_in_leaf': 23, 'max_depth': 10, 'learning_rate': 0.13219411484660287, 'feature_fraction': 0.8, 'colsample_bytree': 0.5}

MAE : 0.46; MSE : 0.38; RMSE : 0.62; R2 : 0.63

<a name="cat-boost"></a>
### Cat Boost
A gradient boosting framework that uses categorical features as input and applies a novel algorithm called Ordered Boosting to reduce overfitting and improve 
prediction accuracy.

Best Parameters: {'subsample': 0.8999999999999999, 'n_estimators': 600, 'max_depth': 9, 'learning_rate': 0.061359072734131756, 'l2_leaf_reg': 54.62277217684348, 'colsample_bylevel': 0.7999999999999999}

MAE : 0.46; MSE : 0.38; RMSE : 0.62; R2 : 0.63

<a name="xg-boost"></a>
### XGBoost
A gradient boosting framework that uses a tree-based learning algorithm and applies several techniques to improve prediction accuracy, such as regularization, parallel processing, and sparsity awareness.

Best Parameters: {'subsample': 0.7999999999999999, 'reg_alpha': 0.016681005372000592, 'n_estimators': 500, 'max_depth': 9, 'learning_rate': 0.018307382802953697, 'gamma': 0.1291549665014884, 'colsample_bytree': 0.6}

MAE : 0.46; MSE : 0.37; RMSE : 0.61; R2 : 0.64

<a name="key-findings"></a>
## Key Findings
XGBoost performed the best among all the models tested, with an R-squared score of 0.64, indicating that 64% of the variance in the target variable can be explained by this model.

<img src="images\cv point plot.png" alt="model-comparison"></img>
The point plot shows that XGBoost had the highest performance, followed by Cat Boost, Light GBM, Gradient Boost and Random Forest while KNN had the lowest performance.

Linear regression, Decision Tree and AdaBoost, performed somewhere in between XGBoost and KNN.

<img src="images\final table.png" alt="table-model-comparison"></img>

The table shows that XGBoost had the lowest mean squared 
error (MSE), root mean squared error (RMSE), and 
mean absolute error (MAE), and the highest R-squared 
score among all the models tested.

<a name="recommendation"></a>
## Recommendations
- Based on the analysis, XGBoost is recommended as the best model for the given dataset and target variable. Further optimization and tuning of XGBoost could potentially improve its performance.
- Feature engineering and selection could be explored to potentially improve the performance of the models.
- Cross-validation techniques such as k-fold or stratified k-fold can be used to validate the model's performance on different subsets of the data and avoid overfitting.

<a name="conclusion"></a>
## Conclusion
- The results of the analysis indicate that XGBoost is the most suitable model for the given dataset and target variable.
- The study demonstrates the importance of exploring multiple models and evaluating their performance to select the best one for the given problem.
- The findings can be used to make data-driven decisions and improve the performance of the model for similar problems in the future.

