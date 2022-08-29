**A New Hybrid Approach to Data-Driven House Prices Predictions**

A house is typically one of the most expensive purchases we make. So before buying a house, there are apparent questions we may ask like, how much approximately a house of my needs may cost, if I am getting a fair price, which locations I can afford, and so on. This project aims to provide an intelligent solution to all these questions using machine learning techniques. 

**Objective:**
There are four key objectives of this project:<br>
1. Estimate the price of a house based on a few basic features such as the number of rooms and bedrooms, square feet, etc.
2. Present a new hybrid approach using two machine learning algorithms to improve prediction.
3. Estimate potential remodeling costs for homeowners.
4. We built a web portal to showcase the functionality of our model.

**Predictive modeling:**
We built five different models for the best house price prediction.<br>
Three Linear Models:<br>
&nbsp; &nbsp; Ridge Regression<br>
&nbsp; &nbsp; Lasso Regression<br>
&nbsp; &nbsp; Elastic Net Regression<br>
One Tree Models:<br>
&nbsp; &nbsp; Random Forest<br>
Our new approach:<br>
&nbsp; &nbsp; Hybrid model<br>
 
**Linear Model:**
 We mentioned earlier that during our data analysis phase, we have seen relationships between multiple variables and sales price, meaning multiple features directly or indirectly impact sales price. Hence we start our first model with Multiple linear regression. We first ensure that our data satisfies or seems to satisfy the following five assumptions:
 1. A linear relationship between the dependent and independent variables
 2. Little or no correlation between independent variables.
 3. The variance of the residuals is constant.
 4. Normal distribution of residuals.
 5. Independence of observations.
 
After feature selection and feature engineering, we choose around 42 features from the initial feature set. We applied multiple linear regression to all 42 selected features. To improve the results, we applied ridge and lasso penalty. For further refinement, we used the elastic net model. Elastic net is a penalized regression model which includes both ridge and lasso penalties.
 

 
**Tree-Based Model:**
Amongst the tree-based model, we used the random forest to explore the predictive accuracy of non-linear models for this problem. Random forest is an estimator algorithm that aggregates the result of many decision trees to output the most optimal result. This model is chosen over other tree-based models because the random forest is robust to outliers and has a lower risk of overfitting. As we mentioned earlier that we have seen linear patterns in the data; therefore, on average linear models perform better than random forest. However, on closer observation of the results, we found in many cases, the sales price prediction of random forest is more accurate than the linear model—that brings us to our new hybrid approach.
 
 

 
**Hybrid Model:**

During our analysis of the linear model and tree-based model, we observed that for some observations, predictions made in linear models are more accurate than in tree-based models, and for other observations, tree-based models are closer to the actual sales price. For example, in the figure below, the horizontal black line signifies all observations. The blue boxes indicate the observations where the linear model performs better, and the green boxes indicate where the tree-based model performs better.  If we can selectively use the model that performs the best for an observation with certain kinds of features, then we can predict the most accurate results. 
To achieve that, we will first need to create a prediction model that will predict which model will perform better for this particular house. Then we use that model to predict the sales price.


![predictiveModel](https://user-images.githubusercontent.com/71456314/187187742-0ac225ab-2a08-49f0-9290-1274b8034e5c.jpg)






Our approach works in the following two phases:

 1. Predict the best prediction model: The aim of this phase is to build a classification model that can predict among linear and tree-based models which will perform the best for an input house. For the linear model, we are using Elastic net, as it is the best among linear models, and random forest as tree based model. To build the classification model, we first need to generate a labeled dataset where we label each observation with the best prediction model. <br>
  a. In order to find the best model, we compute the house price of each observation in the dataset using both linear and random forest models. Then we compare the result of each model with the actual price and create a new column labeled with the most accurate model. For example, the table shown in the figure below shows each observation in the test dataset. The last three columns represent the result by the linear model, result by random forest, and best model, respectively. ‘LM’ and ‘RF’ in the last column Best Model indicates that between the linear model and random forest, which model has the most accurate prediction, where LM indicated the linear model and RF indicated random forest.<br>
  ![hybridmodel2](https://user-images.githubusercontent.com/71456314/187187776-e3730e33-c509-4fb2-8b2a-25b62c508b7d.jpg)

  b. Once we have the labeled dataset, we train a classifier to predict the best model for sales price prediction. We tested SVM, logistic regression, and random forest model and found the random forest classifier to be the most accurate. Therefore we use a random forest classifier to predict the best model between the ‘linear model’ and random forest to predict the sales price for a particular house.

		
 
 2. Use the resultant model to predict sales price: Once we have the best prediction model, we use that model to predict the sales price of that particular house. For each new house step, 1 to 2 is repeated. The figure below showcases the process.

![hybridmodel](https://user-images.githubusercontent.com/71456314/187187035-e99a16c1-925f-4d84-8302-57bc0c0029df.jpg)


Each model pursues the following steps:
1. On the cleaned data, we performed dummy encoding or label encoding of all categorical variables based on the type of categorical variable.
2. We split the dataset into training and testing data, where 80% is the training data, and 20% is the test data. 
3. We randomized the splitting of data by using a random seed. We use 100 random splits for each model; the final result is the average of all 100 training and test datasets.
4. Then, we use the training dataset to fit all of our models and the test dataset to predict the house sales price.
5. The prediction results obtained from all models are then evaluated using R-Squared and root mean squared error. Here R-Squared (R2) determines the proportion of variance in the dependent variable that the independent variables can explain. Root mean squared error (RMSE) measures the error in our prediction. RMSE is the square root of the mean of the square of all of the errors. 


**Results**

The table below shows the result obtained by each model. As we can see Elastic net and Hybrid model both have significantly higher R-squared values compare to the random forest model. The hybrid model is the best model. However, it has only a slightly better R-squared value than Elastic net. 
	According to our assumption, if we are able to predict the correct best prediction model for each house in the test dataset, we will achieve over 95% accuracy. However, our best classifier (random forest) had only a 0.77 R-squared value. That is why we do not see a huge improvement in the results of the hybrid model. By improving the performance of the classifier, we will certainly be able to improve the results of the hybrid model. This is one of our future works that we will explore later.


![result](https://user-images.githubusercontent.com/71456314/187187601-a1516b5b-f75a-4a32-93a9-d6d56e2d8f9a.jpg)


The full repository of the project website can be found here:
https://github.com/jchatterjee/nycdsa_ml_project_website

The fully functioning website can be found here:
https://share.streamlit.io/jchatterjee/nycdsa_ml_project_website/main/trialapp.py
