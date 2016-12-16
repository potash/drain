# run this using the following command:
# drain/bin/drain --outputdir drain/tests/output/ tests.test_drain::calibration
# TODO: add this as an executable test?

from drain import step, model, data
from itertools import product

def prediction():
    # generate the data including a training and test split
    d = data.ClassificationData(n_samples=1000, n_features=100)
    d.target = True

    # construct a random forest estimator
    e = step.Construct(_class_name='sklearn.ensemble.RandomForestClassifier', n_estimators=1)
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
        e = step.Construct(_class_name='sklearn.ensemble.RandomForestClassifier', 
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

        est = step.Construct(_class_name='sklearn.ensemble.RandomForestClassifier',
                n_estimators=n_estimators) 

        fit = model.Fit(inputs=[est, d], return_estimator=True)
        fit.target = True

        predict = model.Predict(inputs=[fit,d])
        predict.target = True

        cal = step.Construct(_class_name='sklearn.calibration.CalibratedClassifierCV', cv=k_folds,
                inputs=[predict], inputs_mapping={'y':None})

        cal_est = model.FitPredict(inputs=[cal, d])
        cal_est.target = True

        metrics = model.PrintMetrics(metrics=[
                {'metric':'baseline'},
                {'metric':'precision', 'k':100},
                {'metric':'precision', 'k':200},
                {'metric':'precision', 'k':300},
        ], inputs=[cal_est])

        steps.append(metrics)

    return steps

def product_model():
    d = data.ClassificationData(n_samples=1000, n_features=100)
    d.target = True

    est = step.Construct(_class_name='sklearn.ensemble.RandomForestClassifier',
                n_estimators=10)
    est.name = 'estimator'

    m1 = model.FitPredict(inputs=[est, d])
    m1.target = True
    m1.name = 'm1'

    m2 = model.FitPredict(inputs=[est, d])
    m2.target = True
    m2.name = 'm2'

    p = model.PredictProduct(inputs=[m1,m2], inputs_mapping=['m1', 'm2'])
    p.target = True
    p.name = 'p'

    return p
