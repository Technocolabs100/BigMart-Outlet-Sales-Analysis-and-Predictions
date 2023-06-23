
## Big-Mart-Sales-Predicition
The aim is to build a predictive model and find out the sales of each product at a particular store. Create a model by which Big Mart can analyse and predict the outlet production sales.

A perfect project to learn Data Analytics and apply Machine Learning algorithms (Linear Regression, Random Forest Regressor, XG Boost) to predict the outlet production sales.
## Dataset Description
BigMart has collected sales data from the year 2013, for 1559 products across 10 stores in different cities. Where the dataset consists of 12 attributes like Item Fat, Item Type, Item MRP, Outlet Type, Item Visibility, Item Weight, Outlet Identifier, Outlet Size, Outlet Establishment Year, Outlet Location Type, Item Identifier and Item Outlet Sales. Out of these attributes response variable is the Item Outlet Sales attribute and remaining attributes are used as the predictor variables.

The data-set is also based on hypotheses of store level and product level. Where store level involves attributes like:- city, population density, store capacity, location, etc and the product level hypotheses involves attributes like:- brand, advertisement, promotional offer, etc.
## Dataset Details
The data has 8523 rows of 12 variables.

Variable - Details
 - Item_Identifier- Unique product ID
 - Item_Weight- Weight of product
 - Item_Fat_Content - Whether the product is low fat or not
 - Item_Visibility - The % of total display area of all products in a store allocated to the particular product
 - Item_Type - The category to which the product belongs
 - Item_MRP - Maximum Retail Price (list price) of the product
 - Outlet_Identifier - Unique store ID
 - Outlet_Establishment_Year- The year in which store was established
 - Outlet_Size - The size of the store in terms of ground area covered
 - Outlet_Location_Type- The type of city in which the store is located
 - Outlet_Type- Whether the outlet is just a grocery store or some sort of supermarket
 - Item_Outlet_Sales - Sales of the product in the particulat store. This is the outcome variable to be predicted.
## Setup
1. pip install jupyter notebook
2. Install python librarys -
 pip install pandas,  pip install numpy,  pip install matplotlib,  pip install klib,  pip install seaborn,  pip install Sklearn,  pip install joblib,  pip install xgboost  pip install flask
## Project Flow
We will handle this problem in a structured way.

1).Problem Statement

2).Hypothesis Generation

3).Loading Packages and Data

4).Data Structure and Content

5).Exploratory Data Analysis

6).Univariate Analysis

7).Bivariate Analysis

8).Missing Value Treatment

9).Feature Engineering

10).Encoding Categorical Variables

11).Label Encoding

12).One Hot Encoding

13).PreProcessing Data

14).Modeling

15).Linear Regression

16).Regularized Linear Regression

17).RandomForest

18).XGBoost

19).Summary