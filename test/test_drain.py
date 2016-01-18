from drain import step, model, data

def calibration():
    steps = []
    for n_estimators in range(100,101):
        d = data.ClassificationData()

        est = step.Construct('sklearn.ensemble.RandomForestClassifier',
                n_estimators=n_estimators, name='estimator')

        holdout = data.HoldOut(p=.25, inputs=[d], inputs_mapping={'index':'train'})

        fit_est = model.FitPredict(inputs=[d, holdout, est], return_estimator=True, 
                inputs_mapping= [{'X':'X', 'y':'y'}, {'train':'index'}, 'estimator'])

        cal = step.Construct('sklearn.calibration.CalibratedClassifierCV', cv='prefit',
                inputs=[fit_est], inputs_mapping= {'base_estimator':'estimator'})

        cal_est = model.FitPredict(inputs=[d, holdout, cal],
                inputs_mapping=[{'X':'X', 'y':'y'}, {'train':'holdout'}, 'estimator'],
                target=True)

        metrics = model.PrintMetrics([
                {'metric':'baseline'},
                {'metric':'precision', 'k':20},
                {'metric':'precision', 'k':30},
                {'metric':'precision', 'k':40},
        ], inputs=[cal_est])

        steps.append(metrics)

    return steps
