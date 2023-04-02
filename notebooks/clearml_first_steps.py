import numpy as np
from sklearn.linear_model import LinearRegression
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

from clearml import Task

task = Task.init(project_name='', task_name='New experiment')

params_dictionary = {'test_param': 123, 'score': -1}

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

task.upload_artifact(name='features', artifact_object=X)
task.upload_artifact(name='targets', artifact_object=y)

reg = LinearRegression().fit(X, y)
joblib.dump(reg, 'model.pkl', compress=True)
params_dictionary['score'] = reg.score(X, y)

task.connect(params_dictionary)