import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression, BayesianRidge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

######################Import Data#############################################################################
rio_df = pd.read_csv('rio_df_clean_for_modeling.csv', index_col='Unnamed: 0')
rio_df.fillna(0, inplace=True)

##############################Instantiate X and y##################################################
X = rio_df.drop(columns=['id', 'price','weekly_price','monthly_price', 'security_deposit','cleaning_fee', 'gender_by_name'])
y = rio_df['price']
log_y = np.log(y)
log_y[y == 0] = 0

#####################Train Test Split and Scale####################################################
X_train, X_test, y_train, y_test = train_test_split(X, log_y, random_state=200)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#########################################Instantiate the Linear Models#############################

lr = LinearRegression()
lasso= LassoCV()
ridge=RidgeCV()
bayes_ridge=BayesianRidge()
sgd=SGDRegressor()
bag=BaggingRegressor() 
random_forrest=RandomForestRegressor() 
ada=AdaBoostRegressor()
gradient_boost=GradientBoostingRegressor()
sv=SVR()
knn=KNeighborsRegressor()
gaus=GaussianProcessRegressor()
dt=DecisionTreeRegressor()

#########################################loop through to show scoring###############################################3#
for model in (lr,lasso,ridge, bayes_ridge, sgd, bag, random_forrest, dt, ada, gradient_boost, sv, knn, gaus):
    print(str(model).split('(')[0].center(40, '-'))
    model.fit(X_train, y_train)
    print(f'Train Score:{model.score(X_train, y_train)}')
    print(f'Test Score:{model.score(X_test, y_test)}')

###############################Check Coefs and dump the unnecessary ones for refitting ############################
coefs = pd.DataFrame([{'column': col, 'coef': val}for val, col in zip(lasso.coef_, X.columns)])
non_zero_cols = list(coefs['column'][coefs['coef'] != 0])

######################################Refit and Scale X #########################################################

X_new=X[non_zero_cols]
X_train, X_test, y_train, y_test = train_test_split(X_new, log_y, random_state=200)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#####################################################Refit and Score Models######################################

for model in (lr, lasso, ridge, bayes_ridge, sgd, bag, random_forrest, dt, ada, gradient_boost, sv, knn, gaus):
    print(str(model).split('(')[0].center(40, '-'))
    model.fit(X_train, y_train)
    print(f'Train Score:{model.score(X_train, y_train)}')
    print(f'Test Score:{model.score(X_test, y_test)}')
