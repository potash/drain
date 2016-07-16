# run this using the following command:
# drain --outputdir drain/test/output/ drain.test.test_drain::calibration
# TODO: add this as an executable test?

from drain import step, model, data
from itertools import product

def prediction():
    # generate the data including a training and test split
    d = data.ClassificationData(target=True, n_samples=1000, n_features=100)

    # construct a random forest estimator
    e = step.Construct('sklearn.ensemble.RandomForestClassifier', n_estimators=1)
    # fit the estimator
    f = model.Fit(inputs=[e, d], return_estimator=True, return_feature_importances=True)
    # make predictions

    p = model.Predict(inputs=[f, d], target=True)
    return p

def n_estimators_search():
    d = data.ClassificationData(target=True, n_samples=1000, n_features=100)
    
    predict = []
    for n_estimators in range(1, 4):
        e = step.Construct('sklearn.ensemble.RandomForestClassifier', n_estimators=n_estimators, name = 'estimator')
        f = model.Fit(inputs=[e, d], return_estimator=True, return_feature_importances=True)

        p = model.Predict(inputs=[f, d], target=True)
        predict.append(p)
        
    return predict

def calibration():
    steps = []
    for n_estimators, k_folds in product(range(50,300,100), [2,5]):
        d = data.ClassificationData(target=True, n_samples=1000, n_features=100)

        est = step.Construct('sklearn.ensemble.RandomForestClassifier',
                n_estimators=n_estimators, name='estimator') 

        fit = model.Fit(inputs=[est, d], return_estimator=True, target=True, name='uncalibrated')
        predict = model.Predict(inputs=[fit,d], target=True, name='y')

        cal = step.Construct('sklearn.calibration.CalibratedClassifierCV', cv=k_folds,
                inputs=[predict], inputs_mapping={'y':None}, name='calibrator')

        cal_est = model.FitPredict(inputs=[cal, d], target=True, name='calibrated')

        metrics = model.PrintMetrics([
                {'metric':'baseline'},
                {'metric':'precision', 'k':100},
                {'metric':'precision', 'k':200},
                {'metric':'precision', 'k':300},
        ], inputs=[cal_est])

        steps.append(metrics)

    return steps

def product_model():
    d = data.ClassificationData(target=True, n_samples=1000, n_features=100)
    est = step.Construct('sklearn.ensemble.RandomForestClassifier',
                n_estimators=10, name='estimator')

    m1 = model.FitPredict(inputs=[est, d], target=True, name='m1')
    m2 = model.FitPredict(inputs=[est, d], target=True, name='m2')

    p = model.PredictProduct(inputs=[m1,m2], target=True, inputs_mapping=['m1', 'm2'], name='p')

    return p
