from drain import step, model, data

def calibration():
    steps = []
    for n_estimators in range(1,10):
        d = data.ClassificationData(target=True)

        est = step.Construct('sklearn.ensemble.RandomForestClassifier',
                n_estimators=n_estimators, name='estimator')

        fit_est = model.FitPredict(inputs=[est, d], name='uncalibrated')

        cal = step.Construct('sklearn.calibration.CalibratedClassifierCV', cv=5,
                inputs=[est], inputs_mapping=['base_estimator'], name='calibrator')

        cal_est = model.FitPredict(inputs=[cal, d], target=True, name='calibrated')

        metrics = model.PrintMetrics([
                {'metric':'baseline'},
                {'metric':'precision', 'k':20},
                {'metric':'precision', 'k':30},
                {'metric':'precision', 'k':40},
        ], inputs=[cal_est], target=True)

        steps.append(metrics)

    return steps
