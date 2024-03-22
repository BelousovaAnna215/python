import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """
    
    model = make_pipeline(StandardScaler(), SVC(kernel='poly', class_weight='balanced', gamma=0.2, degree=2, C=4.7368421052631575))
    model.fit(train_features, train_target)
    
    return model.predict(test_features)
