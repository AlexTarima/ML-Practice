import sys
assert sys.version_info >= (3,7)
from packaging import version
import sklearn
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

# Download the Data

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url,tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

#Loading housing data
housing = load_housing_data()
housing.info()

#Create the test set

import numpy as np
np.random.seed(42)

def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2 ** 32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

#categorizes median_income
housing["income_cat"] = pd.cut(housing["median_income"], bins = [0, 1.5, 3, 4.5, 6., np.inf], labels = [1,2,3,4,5])

from sklearn.model_selection import train_test_split
strat_train_set, strat_test_set = train_test_split(housing, test_size = 0.2, stratify = housing["income_cat"], random_state = 42)

#won't be using income_cat again, so drop
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.drop("median_house_value", axis = 1)

# impute missing values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

# Deal with Categorical values
housing_cat = housing[["ocean_proximity"]]

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

#cat_encoder.handle_unkown = "ignore"
# Otherwise, will raise exception in case of unkown categories

# Transformers

from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer

housing_labels = strat_train_set["median_house_value"].copy()

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self
    
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Clusters {i} similarity" for i in range(self.n_clusters)]

# Pipeline

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector, make_column_transformer

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name)
    )

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)

cluster_simil = ClusterSimilarity(n_clusters=10, gamma = 1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

preprocessing = ColumnTransformer([
    ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
    ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
    ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
    ("geo", cluster_simil, ["latitude", "longitude"]),
    ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
],
remainder=default_num_pipeline
)


# MODELS ###
from sklearn.feature_selection import SelectFromModel
# Evaluate and train data set

# Random Forest Regressor
print("random forest")

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
forest_reg = make_pipeline(preprocessing, RandomForestRegressor())
forest_reg.fit(housing, housing_labels)
forest_rsmes = -cross_val_score(forest_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(forest_rsmes).describe())




# SVM ******************

housing_short = housing.iloc[:5000]
housing_labels_short = housing_labels.iloc[:5000]

print("svm_rbf")
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

svm_rbf_reg = make_pipeline(preprocessing, 
                            SelectFromModel(estimator = forest_reg, threshold = "mean"),
                              SVR(kernel="rbf"))
svm_rbf_reg.fit(housing_short, housing_labels_short)
housing_predictions = svm_rbf_reg.predict(housing_short)
svm_rbf_rmse = mean_squared_error(housing_labels_short, housing_predictions, squared=False)
print(svm_rbf_rmse)


# Testing various hyperparemeters for it
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import loguniform
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
"""
C_range = np.logspace(-6,1,8)
gamma_range = np.logspace(-2,3,6)
param_grid = {
    "svr__C":C_range,
    "svr__gamma":gamma_range
}
grid = GridSearchCV(svm_rbf_reg, param_grid=param_grid)
grid.fit(housing, housing_labels)

print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)

"""
param_distribs={'svr__gamma': loguniform(1e-3, 1e3),
                'svr__C': loguniform(1e-3, 1e4)}

rnd_search = RandomizedSearchCV(svm_rbf_reg, param_distributions=param_distribs, n_iter=16, cv=3, 
                                scoring='neg_root_mean_squared_error', random_state=42)

rnd_search.fit(housing_short, housing_labels_short)
cv_res = pd.DataFrame(rnd_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
print(cv_res)


#Save the model to a file
print("saving the model")

import joblib

joblib.dump(rnd_search.best_estimator_, 'best_svm_model.pkl')

#load the model
"""
print("loading the model")
best_model = joblib.load('best_svm_model.pkl')

predictions = best_model.predict(housing)
print(predictions)
"""












# Linear Regression
print("linear regression")
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)
housing_predictions = lin_reg.predict(housing)

lin_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
print(lin_rmse)

# Decision tree
print("decision tree")
from sklearn.tree import DecisionTreeRegressor
tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)
housing_predictions = tree_reg.predict(housing)
tree_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
print(tree_rmse)

    # Cross validating the decision tree

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(tree_rmses).describe())



# Hyperparameter tuning
print("hyperparameter tuning")
from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
param_grid = [
    {'preprocessing__geo__n_clusters':[5,8,10],
     'random_forest__max_features':[4,6,8]},
    {'preprocessing__geo__n_clusters':[10,15],
     'random_forest__max_features':[6,8,10]}
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)

print(grid_search.best_params_)

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res.head()



