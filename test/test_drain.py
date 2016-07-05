# run this using the following command:
# drain --outputdir drain/test/output/ drain.test.test_drain::calibration
# TODO: add this as an executable test?

from drain import step, model, data
from itertools import product

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
    d = data.ClassificationData(target=True, n_samples=1000, n_features=100)
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
