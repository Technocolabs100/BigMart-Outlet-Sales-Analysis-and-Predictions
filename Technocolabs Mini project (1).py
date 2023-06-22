#!/usr/bin/env python
# coding: utf-8

# # Data Interpretation

# In[1]:


import os #paths to file
import numpy as np # linear algebra
import pandas as pd # data processing
import warnings  # warning filter
import matplotlib.pyplot as plt # Data visulization
import seaborn as sns # Data visulization
get_ipython().run_line_magic('matplotlib', 'inline')


#train test split
from sklearn.model_selection import train_test_split

#feature engineering
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#metrics
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2 
from sklearn.model_selection  import cross_val_score as CVS

#ML models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


# Cross Validation
from sklearn.model_selection import cross_val_score as CVS
import pandas as pd
df =pd.read_csv(r"C:/Users/hp/Downloads/9961_14084_bundle_archive/Test.csv")
df


# # problem statement

# In[2]:


#The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. 
#Also, certain attributes of each product and store have been defined.
#The aim of this data science project is to build a predictive model and find out the sales of each product at a particular store.


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull()


# # Data Loading

# In[6]:


import pandas as pd
df=pd.read_csv("C:/Users/hp/Downloads/9961_14084_bundle_archive/Train.csv")
df
df.head()


# In[7]:


df.info()


# # Data combining

# In[8]:


# test_df['Item_Outlet_Sales'] = np.nan
df['source'] = 'df'
df['source'] = 'df'
data = pd.concat([df, df], ignore_index=True)
print('After Combining Datasets: ', data.shape)


# In[9]:


df.info(verbose=True,show_counts=True)


# ### Data exploration
# 

# #### Missing values
# #### variable identification
# #### Univariate Analysis
# #### Bi-variate Analysis
# #### Outlier treatment
# #### Variable transformation
# #### Variable creation

# # Missing value treatment

# In[10]:


# Lets check missing Values
print('Test:\n')
print('Missing Values by Count: \n\n',
      data.isnull().sum().sort_values(ascending=False),'\n\nMissing Values by %:\n\n',
      data.isnull().sum().sort_values(ascending=False)/data.shape[0] * 100)


# In[11]:


#Lets check and imputate missing value
print('Missing Values in Outlet_Size:\n\n',df.Outlet_Size.value_counts())


# In[12]:


print('\missing values in Item_Weight:\n\n',df.Item_Weight.value_counts())


# # Outlet_size is a categorical column, we can use mode to fill values

# In[13]:


# Filling outlet missing values.
# data['Outlet_size']= data.Outlet_Size.fillna(data.Outlet_Size,dropna().mode()[0])
# Checking if we filled all values

print('Missing Values in Item_Weight: ', len(df[df.Outlet_Size.isnull()]))
miss_values = data.Outlet_Size.isnull()
O_Size_avg = data.pivot_table(values='Outlet_Size', index='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
data.loc[miss_values, 'Outlet_Size'] = data.loc[miss_values, 'Outlet_Type'].apply(lambda x:O_Size_avg.loc[x])
print('Missing values after filling: ' , sum(data['Outlet_Size'].isnull()))


# In[14]:



sns.pairplot(df)


# In[15]:


sns.boxplot(x=data['Item_Weight'], palette='BuPu')
plt.title('Item Wieght Distribution')


# # check info for missing values
# data.info()

# #  EDA Analysis

# In[16]:


# Best Auto EDA Analysis
# data1 = data.drop(["Item_Identifier", "Outlet_Identifier"], axis=1)
# Data_Profile = ProfileReport(data1)
# Data_Profile


# # Variable identication

# In[17]:


# Numericals
num_df = data.select_dtypes('number')
# Categorial 
cat_df = data.select_dtypes('object')


# In[18]:


#lets deal with the categorical data first
for col in cat_df.columns:
    if(col !='Item_Identifier'):
        print('\nfrequency of Categories for varible %s'%col)
        print('\ntotal Categories:', len(cat_df[col].value_counts()), '\n',cat_df[col].value_counts())


# ## Item_Fat_Content: We have reapted values in , lets replace them
# ## Item_Type: We have categories of items, that can be shrink
# ## Outlet_Type: We have Store type2,and type3, that can be combined

# In[19]:


data['Item_Fat_Content'] = data.Item_Fat_Content.replace(['LF', 'low fat', 'reg'],
                                                              ['Low Fat','Low Fat', 'Regular'])
data.Item_Fat_Content.value_counts()


# In[20]:


# Combine Item_Type, and create new category
data['Item_Type_Combined'] = data.Item_Identifier.apply(lambda x:x[0:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].replace(['FD','DR','NC'],
                                                                   ['Food','Drinks', 'Non-Consumable']) 
data.Item_Type_Combined.value_counts()


# In[21]:


data.pivot_table(values='Item_Outlet_Sales', index='Outlet_Type')


# There is a huge difference in sales, so not good idea to combine them

# In[22]:


# Lets deal with Numerical Data
num_df.describe()


# # outliers 

# In[23]:


#Box plot for Item_Outlet_Sales to see outliers
sns.boxplot(x=df['Item_Outlet_Sales'], palette='BuPu')
plt.title('Item Outlet Sales Distribution')


# In[24]:


# Removing Outliers
def outliers(df, feature):
    Q1= df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    return upper_limit, lower_limit

upper, lower = outliers(data, "Item_Outlet_Sales")
print("Upper whisker: ",upper)
print("Lower Whisker: ",lower)
data = data[(data['Item_Outlet_Sales'] > lower) & (data['Item_Outlet_Sales'] < upper)]


# In[25]:


# Item_Outlet_Sales after removing Outliers
sns.boxplot(x=data['Item_Outlet_Sales'], palette='BuPu')
plt.title('Item Outlet Sales Distribution after removing outliers')


# In[26]:


data.head()


# # Data vizualization

# In[27]:


# Categorical data

# COUNTPLOT FOR ITEM FAT_CONTENT
plt.figure(figsize=(6,4))
sns.countplot(data=data, x='Item_Fat_Content')
plt.xlabel('Item Fat Content', fontsize=15)
plt.show()


# In[28]:


# CountPlot for Individual Item Category
plt.figure(figsize=(24,6))
sns.countplot(data=data, x='Item_Type',  palette='Set1')
plt.xlabel('Individual Item Category ', fontsize=30)
plt.show()


# In[29]:


# CountPlot for Outlet_Identifier
plt.figure(figsize=(15,4))
sns.countplot(data=data, x='Outlet_Identifier', palette='winter')
plt.xlabel('Stores', fontsize=15)
plt.show()


# In[30]:


# CountPlot for Outlet_Size
plt.figure(figsize=(6,4))
sns.countplot(data=data, x='Outlet_Size')
plt.xlabel('Store Size', fontsize=15)
plt.show()


# In[31]:



# CountPlot for Outlet_Location_Type
plt.figure(figsize=(10,4))
sns.countplot(data=data, x='Outlet_Location_Type')
plt.xlabel('Store Location Type', fontsize=15)
plt.show()


# In[32]:


#Countplot for outlet type
plt.figure(figsize=(10,7))
sns.countplot(data=data, x='Outlet_Type')
plt.xlabel('Store Type',fontsize=15)
plt.show()


# In[33]:


# CountPlot for Outlet_Type
plt.figure(figsize=(10,7))
sns.countplot(data=data, x='Outlet_Type')
plt.xlabel('Store Type', fontsize=15)
plt.show()


# In[34]:


plt.figure(figsize=(60,20))
plot=sns.barplot(x='Item_Weight',y='Item_Visibility',data=df)


# In[35]:


df.columns


# In[36]:


# For Numerical Data

# HistPlot for Outlet_Age
plt.hist(x=data['Item_Outlet_Sales'])
plt.title('Store Age')
plt.show()


# # Bivariate plots for Numerical

# In[37]:


#Scatter plots for Sales Per Item_Visiblity
plt.scatter(df['Item_Visibility'],df['Item_Outlet_Sales'])
plt.title('Sales based on item visiblity')
plt.xlabel('Item_Visibility')
plt.ylabel('Item_Outlet_Sales')
plt.show()


# In[38]:


#Scatter plot for sales per item _MRP
import matplotlib.pyplot as plt
plt.scatter(df['Item_MRP'],df['Item_Outlet_Sales'])
plt.title('SALES BASED ON ITEM MRP')
plt.xlabel('Item_MRP')
plt.ylabel('Item_Outlet_Sales')
plt.show()


# # Bivariate plots for categorical values
# lets Check For the following relationship
# ## Sales per Item_Type_Combined
# ## Sales per Outlet_Identifier
# ## Sales per Outlet_Type
# ## Sales per Outlet_Size
# ## Sales per Outlet_Location_Type

# In[39]:


# Barplot for sales per Item_Type
plt.figure(figsize=(28,6))
sns.barplot(data=data,x='Item_Type', y='Item_Outlet_Sales',palette='flag')
plt.title('Sales based on individual item category',fontsize=30)
plt.xlabel('Individual Item Category', fontsize=40)
plt.ylabel('Sales',fontsize=25)
plt.legend()
plt.show()


# In[40]:


df.hist(bins=25,figsize=(20,10))


# In[41]:


# BarPlot for Sales per Item_Type_Combined
plt.figure(figsize=(10,4))
sns.barplot(data=data,x='Item_Type', y='Item_Outlet_Sales')
plt.title('Sales based on Item Category')
plt.xlabel('Item Category ')
plt.ylabel('Sales')
plt.show()


# In[42]:


#Barplot for the sales per outlet_Identifier
plt.figure(figsize=(10,4))
sns.barplot(data=data,x='Outlet_Establishment_Year', y='Item_Outlet_Sales')
plt.title('Sales based on Stores')
plt.xlabel("Year")
plt.ylabel('Sales')
plt.show()


# In[43]:


#Barplot for the Sales per Outlet_Type
plt.figure(figsize=(8,5))
sns.barplot(data=data,x='Outlet_Type',y='Item_Outlet_Sales')
plt.title("Sales based on Store type")
plt.xlabel("Stpre type")
plt.ylabel("Sales")
plt.legend()
plt.show()


# In[44]:


plt.plot(df['Item_Identifier'].value_counts().sort_index())
sns.pairplot(df)


# In[45]:


sns.set(style="whitegrid")
sns.pairplot(df, hue="Outlet_Type")


# In[46]:


sns.distplot(df['Item_Outlet_Sales'],kde=True)


# In[47]:


# BarPlot for Sales per Outlet_Location_Type
plt.figure(figsize=(10,4))
sns.barplot(data=data,x='Outlet_Location_Type', y='Item_Outlet_Sales')
plt.title('Sales based on Store location type ')
plt.xlabel('Store location type')
plt.ylabel('Sales')
plt.show()


# In[48]:


plt.figure(figsize=(10,5))
sns.barplot(data=data,x='Outlet_Location_Type', y='Item_Outlet_Sales',hue='Outlet_Type',palette='magma')
plt.title('Sales based on Store location type ')
plt.xlabel('Store location type')
plt.ylabel('Sales')
plt.legend()
plt.show()


# In[49]:


plt.figure(figsize=(15,15))
sns.barplot(data=data,x='Item_Outlet_Sales',y='Outlet_Type',hue='Item_Type',palette='Set1')
plt.title('Individual items in each store',fontsize=20)
plt.xlabel('sales',fontsize=15)
plt.ylabel('store Type',fontsize=15)
plt.yticks(rotation=80)
plt.show()


# In[50]:


plt.figure(figsize=(10,5))
sns.barplot('Outlet_Location_Type','Item_Outlet_Sales',hue='Outlet_Type',data=data,palette='magma')
plt.legend()
plt.show()
import warnings
warnings.filterwarnings("ignore")


# In[51]:


plt.figure(figsize=(10,5))
sns.barplot(data=data,x='Outlet_Location_Type',y='Item_Outlet_Sales',hue='Item_Type',palette='Set1')
plt.xlabel('items category in each store', fontsize=10)
plt.ylabel('Sales', fontsize=10)
plt.legend()
plt.show()


# In[52]:


plt.figure(figsize=(10,5))
sns.barplot('Outlet_Type','Item_Outlet_Sales',hue='Outlet_Location_Type',data=data,palette='magma')
plt.legend()


# 1)Seafood is the most item_type sold in SuperMarket 1 and 2, Grocery store has less sales.
# 2)Only Teir3 has all Outlet_Type, and SuperMarket type3 has most sales..
# 3)Outlet_Location_Type has almost equal sales based on Item_Type_combined.

# In[53]:


# Correlation Matrix
plt.Figure(figsize=(20,5))
sns.heatmap(data.corr(), annot=True)


# In[54]:


plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),annot=True,cmap="gray_r")


# # feature engineering 
# 

# 1)categorical encoding
# 2)variable transformation
# 3)outlier engineering
# 4)Date and time engineering
# 

# # Label Encoding

# In[55]:


#Label Encoding for Ordinal Data
le = LabelEncoder()
label = ['Item_Identifier','Item_Fat_Content','Item_Type', 'Outlet_Type', 'Outlet_Location_Type', 'Outlet_Size']
for i in label:
    data[i] = le.fit_transform(data[i])
data.head()


# In[56]:


df.describe()


# In[57]:


df.head(5)


# One-hot encoding in machine learning is the conversion of categorical information into a format that may be fed into machine learning algorithms to improve prediction accuracy.

# # One Hot Encoding

# In[58]:


#One hot encoding for Nominal data
#column for applying the one hot encoding
from sklearn.preprocessing import OneHotEncoder
cols=['Item_Type']
#Appling the One hot Encoder
OH_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
data_oh = pd.DataFrame(OH_encoder.fit_transform(data[cols])).astype('int64')

#get feature columns
data_oh.columns = OH_encoder.get_feature_names_out(cols)
# one hot encoding removed index;put it back
data_oh.index= data.index
# # # Add one-hot encoded columns to our main df new name: tr_fe, te_fe (means feature engeenired) 
data_fe = pd.concat([data, data_oh], axis=1)
data_fe.head()


# In[59]:


#One Hot Coding:
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                              'Item_Type'])


# In[60]:


data_fe.head()


# In[61]:


plt.figure(figsize=(10,9))
sns.heatmap(df.corr(),  annot=True)
['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
       'Item_Type', 'Item_MRP', 'Outlet_Identifier',
       'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type', 'Item_Outlet_Sales', 'source']


# In[ ]:





# In[62]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (12,6))
ax = sns.boxplot(x = 'Item_Fat_Content', y = 'Item_Outlet_Sales', data = df)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
ax.set_title('Outlet years vs Item_Outlet_Sales')
ax.set_xlabel('', fontsize = 15)
ax.set_ylabel('Item_Outlet_Sales', fontsize = 15)

plt.show()


# In[63]:


df.index = df['Outlet_Establishment_Year']
df = df.loc[:,['Item_Outlet_Sales']]
ts = df['Item_Outlet_Sales']
plt.figure(figsize=(12,8))
plt.plot(ts, label='Item_Outlet_Sales')
plt.title('Outlet Establishment Year')
plt.xlabel('Time(year-month)')
plt.ylabel('Item_Outlet_Sales')
plt.legend(loc = 'best')
plt.show()


# In[64]:


data.groupby('Outlet_Establishment_Year')['Item_Outlet_Sales'].mean().plot.bar()


# In[65]:


sns.distplot(data['Item_Outlet_Sales'])


# In[95]:



plt.figure(figsize = (14,9))

plt.subplot(211)
ax = sns.boxplot(x='Outlet_Identifier', y='Item_Outlet_Sales', data=data, palette="Set1")
ax.set_title("Outlet_Identifier vs. Item_Outlet_Sales", fontsize=15)
ax.set_xlabel("", fontsize=12)
ax.set_ylabel("Item_Outlet_Sales", fontsize=12)




plt.show()


# In[67]:


data.dtypes


# # 2) Splitting the dataset into train and test

# # Data preprocessing

# In[68]:


X=df.drop('Item_Outlet_Sales',axis=1)


# In[69]:


Y=df['Item_Outlet_Sales']


# In[70]:


from sklearn.model_selection import train_test_split

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, random_state=101)


# In[71]:


X_Train.shape


# In[72]:


X_Test.shape


# In[73]:


X=df.drop("Item_Outlet_Sales",axis=1)
y=df["Item_Outlet_Sales"]
X_Train,X_Test,y_Train,y_Test=train_test_split(X,y,train_size=0.700,random_state=100)


# In[74]:


X_Train.shape


# In[75]:


X_Test.shape


# In[76]:


df.columns


# In[77]:


X_Test.head()


# In[78]:


# Drop irrlevent Columns
data_fe = data_fe.drop(['Item_Identifier', 'Outlet_Identifier',
                     'Item_Fat_Content','Item_Visibility', 'Item_Type',
                     'Outlet_Establishment_Year','Item_Type'], axis=1)
data_fe.head()

# Divide Data into train and test
train = data_fe.loc[data_fe['source']=="train"]
test = data_fe.loc[data_fe['source']=="test"]

train = train.drop('source', axis=1)
test = test.drop(['source',  'Item_Outlet_Sales'], axis=1)
# Check Datasets
print('\nTrain Dataset for Model Buidling: \n')
print(train.info(verbose=True, show_counts=True))
print('\nTest Dataset for Model Buidling: \n')
print(test.info(verbose=True, show_counts=True)) 
train.head()


# In[79]:


# Train and Test split
X=df.drop("Item_Outlet_Sales",axis=1)
y=df["Item_Outlet_Sales"]
X_Train,X_Test,y_Train,y_Test=train_test_split(X,y,train_size=0.700,random_state=100)
def cross_val(model, X, y, cv):
    scores = CVS(model, X, y, cv=cv)
    print(f'{model} Scores:')
    for i in scores:
        print(round(i,2))
    print(f'Average {model} score: {round(scores.mean(),4)}')
    


# In[80]:


from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
pipeline_lr = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="-1")),
        ("VarianceThreshold", VarianceThreshold(threshold=0.1)),
        ("scalar1", StandardScaler()),
        ("pca1", PCA(n_components=2)),
        ("lr_classifier", LogisticRegression(random_state=0)),
    ]
)


pipeline_dt = Pipeline(
    [
        ("scalar2", StandardScaler()),
        ("VarianceThreshold", VarianceThreshold(threshold=0.1)),
        ("imputer", SimpleImputer(strategy="constant", fill_value="-1")),
        ("pca2", PCA(n_components=2)),
        ("dt_classifier", DecisionTreeClassifier()),
    ]
)

pipeline_randomforest = Pipeline(
    [
        ("scalar3", StandardScaler()),
        ("pca3", PCA(n_components=2)),
        ("rf_classifier", RandomForestClassifier()),
    ]
)


# # Data modeling

# In[81]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0)
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
# The pipeline can be used as any other estimator # and avoids leaking the test set into the train set
pipe.fit(X_train, y_train)
Pipeline(steps=[('scaler', StandardScaler()), ('svc', SVC())])
pipe.score(X_test, y_test)


# # pipeline to link all steps of data manupulation together to create pipeline

# # LINEAR REGRESSION

# In[82]:


# Model
import warnings
warnings.filterwarnings("ignore")
model = LinearRegression(normalize=True)

# Fit
model.fit(X_train, y_train)

# Predict
y_predict = model.predict(X_test)

# Score Metrics for Regression:
LR_MAE = MAE(y_test, y_predict)
LR_MSE = MSE(y_test, y_predict)
LR_R_2 = R2(y_test, y_predict)
print(f" Mean Absolute Error: {LR_MAE}\n")
print(f" Squared Mean Squared Error: {np.sqrt(LR_MSE)}\n")
print(f" R^2 Score: {LR_R_2}\n")

# Cross Validation Score check
cross_val(LinearRegression(),X,y,5)


# # LASSO REGRESSION

# In[83]:


# Model
import warnings
warnings.filterwarnings("ignore")
model = Lasso(alpha=0.05, normalize=True)

# Fit
model.fit(X_train, y_train)

# Predict
y_predict = model.predict(X_test)

# Score Metrics for Regression:
LS_MAE = MAE(y_test, y_predict)
LS_MSE = MSE(y_test, y_predict)
LS_R_2 = R2(y_test, y_predict)
print(f" Mean Absolute Error: {LS_MAE}\n")
print(f" Mean Squared Error: {LS_MSE}\n")
print(f" R^2 Score: {LS_R_2}\n")

# Cross Validation Score check
cross_val(Lasso(),X,y,5)


# # DECISION TREE REGRESSOR

# In[84]:


# Model
model = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)

# Fit
model.fit(X_train, y_train)

# Predict
y_predict = model.predict(X_test)

# Score Metrics (This is where it is not complete)
DR_MAE = MAE(y_test, y_predict)
DR_MSE = MSE(y_test, y_predict)
DR_R_2 = R2(y_test, y_predict)
print(f" Mean Absolute Error: {DR_MAE}\n")
print(f" Mean Squared Error: {DR_MSE}\n")
print(f" R^2 Score: {DR_R_2}\n")

# Cross Validation Score check
cross_val(DecisionTreeRegressor(),X,y,5)


# # RANDOM FOREST REGRESSOR

# In[85]:


# Model
model = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=100, n_jobs=4, random_state=101)

# Fit
model.fit(X_train, y_train)

# Predict
y_predict = model.predict(X_test)

# Score Matrics
RFR_MAE = MAE(y_test, y_predict)
RFR_MSE = MSE(y_test, y_predict)
RFR_R_2 = R2(y_test, y_predict)
print(f" Mean Absolute Error: {RFR_MAE}\n")
print(f" Mean Squared Error: {RFR_MSE}\n")
print(f" R^2 Score: {RFR_R_2}\n")
cross_val(RandomForestRegressor(),X, y, 5)


# In[86]:


pip install xgboost


# # XGBOOST

# In[87]:


# Model
from xgboost import XGBRegressor
model = XGBRegressor()

# Fit
model.fit(X_train, y_train)

# Predict
y_predict = model.predict(X_test)

# Score Matrix
XG_MAE = MAE(y_test, y_predict)
XG_MSE = MSE(y_test, y_predict)
XG_R_2 = R2(y_test, y_predict)
print(f" Mean Absolute Error: {XG_MAE}\n")
print(f" Mean Squared Error: {XG_MSE}\n")
print(f" R^2 Score: {XG_R_2}\n")
cross_val(XGBRegressor(),X, y, 5)


# # Support Vector Regressor

# In[88]:


# Model
model = SVR()

# Fit
model.fit(X_train, y_train)

# Predict
y_predict = model.predict(X_test)

# Score Matrics
SVR_MAE = MAE(y_test, y_predict)
SVR_MSE = MSE(y_test, y_predict)
SVR_R_2 = R2(y_test, y_predict)
print(f" Mean Absolute Error: {SVR_MAE}\n")
print(f" Mean Squared Error: {SVR_MSE}\n")
print(f" R^2 Score: {SVR_R_2}\n")
cross_val(SVR(),X, y, 5)


# # KNN REGRESSOR

# In[89]:


model = KNeighborsRegressor(n_neighbors=7)

# Fit
model.fit(X_train, y_train)

# Predict
y_predict = model.predict(X_test)

# Score Matrics
KNR_MAE = MAE(y_test, y_predict)
KNR_MSE = MSE(y_test, y_predict)
KNR_R_2 = R2(y_test, y_predict)
print(f" Mean Absolute Error: {KNR_MAE}\n")
print(f" Mean Squared Error: {KNR_MSE}\n")
print(f" R^2 Score: {KNR_R_2}\n")
cross_val(KNeighborsRegressor(),X, y, 5)


# # ADA Boost Regressor

# In[90]:


model = AdaBoostRegressor()

# Fit
model.fit(X_train, y_train)

# Predict
y_predict = model.predict(X_test)

# Score Matrics
AB_MAE = MAE(y_test, y_predict)
AB_MSE = MSE(y_test, y_predict)
AB_R_2 = R2(y_test, y_predict)
print(f" Mean Absolute Error: {AB_MAE}\n")
print(f" Mean Squared Error: {AB_MSE}\n")
print(f" R^2 Score: {AB_R_2}\n")
cross_val(AdaBoostRegressor(),X, y, 5)


# # Summary

# # Using this model Big Mart will try to understand the properties of the products and the stores which play a key role in increasing role

# In[ ]:





# In[ ]:




