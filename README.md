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
7. [Fitting Models to the Data](model-fitting)
    - [Logistic Regression](#log-reg)
    - [Decision Tree](#dt)
    - [Random Forest](#rf)
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

The shape of the dataset is : (5630, 20)

During the data preprocessing phase, there were some inconsistencies, such as duplicate entries in categorical columns. For instance, categories like "mobile phone" and "phone" referred to the same item but were recorded under different labels. These duplicates were consolidated to ensure uniformity across the dataset.

Additionally, approximately 5% of missing data was detected in 4 to 5 columns. To address this, we applied **K-Nearest Neighbors (KNN)** Imputation techniques to replace the missing values, ensuring that the dataset maintained its integrity for further analysis. This approach helped preserve the underlying structure of the data while mitigating the impact of missing information.


<a name="phase-2"></a>
### Phase II - Exploratory Data Analysis
- Performing univariate, bivariate and multivariate analysis to understand the data
- Creating visualizations to summarize and present the data
- Calculating summary statistics such as mean, median and standard deviation to describe the data

<br>

<img src="images\single_pie.png" alt="percentage-of-customers-who-left"></img>

**Customer Satisfaction**: The high retention rate suggests that most customers are satisfied with the product or service. This could be due to effective customer service, high product quality, or competitive pricing.

**Areas for Improvement**: The 16.84% attrition rate indicates that there are still some customers who are not fully satisfied. Identifying the reasons behind their departure can help in formulating strategies to reduce this percentage.

<br><br>

<img src="images\multiple_pie.png" alt="churn-percentage-by-gender"></img>

**Customer Satisfaction**: Both genders show high retention rates, indicating overall customer satisfaction. However, female customers appear to be slightly more satisfied or loyal.

**Targeted Strategies**: The higher attrition rate among male customers suggests a need for targeted strategies to improve their retention.

<br><br>

<img src="images\churn-tenure.png" alt="churn-by-tenure-range"></img>

**Early Engagement**: The high churn rate in the first 18 months highlights the need for improved engagement and support for new customers. Ensuring a positive onboarding experience and addressing any initial issues promptly can help reduce early churn.

**Loyalty Programs**: The lower churn rates in the 18-30 and 30+ month ranges suggest that customers who stay longer are more loyal. Implementing loyalty programs and incentives for long-term customers can further enhance retention.

**Targeted Interventions**: Identifying the specific reasons for early churn and addressing them through targeted interventions can help improve overall retention rates.

<br><br>

<img src="images\line.png" alt="estimated-monthly-spending-by-tenure"></img>

**Early Engagement**: The high variability in spending among new customers highlights the importance of effective onboarding and engagement strategies to stabilize and increase spending.

**Mid-Tenure Challenges**: The drop in spending around 30 months suggests potential challenges in maintaining customer interest and satisfaction during this period. Identifying and addressing these challenges can help sustain spending levels.

**Long-Term Engagement**: The fluctuations in long-term spending indicate that even loyal customers may have periods of disengagement. Regularly refreshing engagement strategies can help maintain consistent spending.

<br><br>

<img src="images\churn-ss.png" alt="churn-by-satisfaction-score"></img>

**Retention Despite Low Satisfaction**: The high retention rates among customers with low satisfaction scores could indicate that these customers might be staying due to lack of alternatives or other factors not directly related to satisfaction.

**Importance of Satisfaction**: The trend of decreasing churn with higher satisfaction scores underscores the importance of maintaining high customer satisfaction to reduce churn rates.

<br><br>

<img src="images\box.png" alt="boxen-plot-city-tier"></img>

**Customer Behavior by City Tier**: Customers in higher city tiers (Tier 1) exhibit more variability in their spending behavior, possibly due to diverse economic conditions and spending power.

**Satisfaction and Spending Stability**: Higher satisfaction scores are linked to more stable spending patterns, indicating that satisfied customers are less likely to react drastically to order amount hikes.

<br><br>

<img src="images\pairplot.png" alt="correlation-plot"></img>

**Targeted Retention Strategies**: Focus on customers with shorter tenures and higher monthly charges for retention efforts. Personalized offers or discounts could help reduce churn in these segments.

**Demographic-Specific Interventions**: Develop targeted strategies for senior citizens and customers without partners or dependents to address their specific needs and reduce churn.

**Service Optimization**: Evaluate the impact of phone service and multiple lines on customer satisfaction and churn. Consider offering bundled services or incentives to customers with multiple lines to enhance retention.


<a name="model-fitting"></a>
## Fitting Models to the Data

The train-test split method was used to evaluate the performance of machine learning models. This method involves splitting the available dataset into two 
parts: a training set and a testing set. The training set,which accounted for 80% of the data, was used to train the machine learning models, while the remaining 20% was used for testing the models. The training set had 4,504 records, and the testing set had 1,126 records. The train-test split allowed for the evaluation of the machine learning models on new, unseen data, which is essential for determining their effectiveness and generalizability.

<a name="log-reg"></a>
### Logistic Regression
A statistical model that predicts the probability of a binary outcome (such as success/failure, yes/no) based on one or more independent variables. 

**Classification Report**


                   precision    recall  f1-score   support

              0       0.91      0.96      0.93       938
              1       0.73      0.50      0.59       188

    accuracy                              0.89      1126
    macro avg         0.82      0.73      0.76      1126
    weighted avg      0.88      0.89      0.88      1126

**Confusion Matrix**

<img src="images\lg_cf.png" alt="log-reg-confusion-matrix"></img>

<a name="dt"></a>
### Decision Tree
A tree-structured model that breaks down a dataset into smaller and smaller subsets based on a set of decisions or rules until the subsets contain instances with a single class or value.

**Classification Report**

                  precision    recall  f1-score   support

             0       0.97      0.97      0.97       938
             1       0.86      0.86      0.86       188

    accuracy                             0.95      1126
    macro avg        0.91      0.92      0.92      1126
    weighted avg     0.95      0.95      0.95      1126

**Confusion Matrix**

<img src="images\dt_cf.png" alt="dt-confusion-matrix"></img>


<a name="rf"></a>
### Random Forest
An ensemble model that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

**Classification Report**

                  precision    recall  f1-score   support

             0       0.96      1.00      0.98       938
             1       0.97      0.81      0.89       188

    accuracy                             0.97      1126
    macro avg        0.97      0.90      0.93      1126
    weighted avg     0.97      0.97      0.96      1126

**Confusion Matrix**

<img src="images\rf_cf.png" alt="rf-confusion-matrix"></img>

<a name="key-findings"></a>
## Key Findings
Random Forest performed the best among all the models tested, with a better train accuracy as well as test accuracy.

<a name="recommendation"></a>
## Recommendations
- During our analysis we found that most customers are leaving at the first few months of the service tenure. These customers churned maybe because they found a better alternative to what the company was already offering. A better understanding of the competitora as well as providing attractive offers to new customers can be a way to improve retention.
- Actively seeking feedback from customers who leave early as well as concentrating on providing flexible plans to cities where the customer satisfaction scores are low can be also implemented to reduce churn.
- Implementing regular follow-up to customers who recently joined can improve their perspective of the company as well as motivate them to refer others.

<a name="conclusion"></a>
## Conclusion
- In conclusion, the E-commerce Churn Analysis project successfully developed a machine learning model capable of predicting customer churn with high accuracy. By leveraging various features such as customer demographics, purchasing behavior, and engagement metrics, the model provides valuable insights into the factors driving customer attrition.

- The use of advanced data preprocessing techniques, including the handling of duplicate entries and missing values through KNN Imputation, ensured the dataset’s quality and reliability. The findings from this project can help e-commerce platforms implement targeted retention strategies, improve customer experiences, and ultimately reduce churn rates.

- This analysis not only offers immediate business value but also contributes to a broader understanding of customer retention in the e-commerce sector. Future research can further refine these models and explore additional factors influencing churn, enhancing the effectiveness of data-driven decision-making in this dynamic industry.
