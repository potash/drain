# run this using the following command:
# drain/bin/drain execute --path drain/tests/output/ -w tests.test_drain::calibration

from drain import step, model, data
from drain.step import MapResults
from itertools import product
from sklearn import ensemble

def prediction(n_samples=1000, n_features=100):
    # generate the data including a training and test split
    d = data.ClassificationData(n_samples=n_samples, n_features=n_features)
    d.target = True

    # construct a random forest estimator
    e = step.Call(ensemble, "RandomForestClassifier", n_estimators=1)
    e.target = False

    # fit the estimator
    f = model.Fit(inputs=[e, d], return_estimator=True, return_feature_importances=True)

    # make predictions
    p = model.Predict(inputs=[f, d])
    p.target = True
    return p

def n_estimators_search():
    d = data.ClassificationData(n_samples=1000, n_features=100)
    d.target = True
    
    predict = []
    for n_estimators in range(1, 4):
        e = step.Call(ensemble, 'RandomForestClassifier', 
                n_estimators=n_estimators)
        f = model.Fit(inputs=[e, d], return_estimator=True, return_feature_importances=True)

        p = model.Predict(inputs=[f, d])
        p.target = True
        predict.append(p)
        
    return predict

def calibration():
    steps = []
    for n_estimators, k_folds in product(range(50,300,100), [2,5]):
        d = data.ClassificationData(n_samples=1000, n_features=100)
        d.target = True

        est = step.Call(ensemble, 'RandomForestClassifier',
                n_estimators=n_estimators) 

        fit = model.Fit(inputs=[est, d], return_estimator=True)
        fit.target = True

        predict = model.Predict(inputs=[fit,d])
        predict.target = True

        cal = step.Call('sklearn.calibration.CalibratedClassifierCV', cv=k_folds,
                inputs=[MapResults([predict], {'y':None})])

        cal_est = model.FitPredict(inputs=[cal, d])
        cal_est.target = True

        steps.append(cal_est)

    return steps

def product_model():
    d = data.ClassificationData(n_samples=1000, n_features=100)
    d.target = True

    est = step.Call(ensemble, 'RandomForestClassifier',
                n_estimators=10)
    est.name = 'estimator'

    m1 = model.FitPredict(inputs=[est, d])
    m1.target = True
    m1.name = 'm1'

    m2 = model.FitPredict(inputs=[est, d])
    m2.target = True
    m2.name = 'm2'

    p = model.PredictProduct(inputs=[MapResults([m1,m2], ['m1', 'm2'])])
    p.target = True
    p.name = 'p'

    return p
