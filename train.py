import os
import warnings
import sys
import click

import mlflow
import mlflow.sklearn

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

@click.command()
@click.option("--run-name", type=click.STRING, default="run-name")
@click.option("--n-estimators", type=click.INT, default=500)
@click.option("--max-depth", type=click.INT, default=4)
@click.option("--min-samples-split", type=click.INT, default=2)
@click.option("--learning-rate", type=click.FLOAT, default=0.01)
@click.option("--show-graph", type=click.BOOL, default=False)
def mlflow_run(run_name="take1", n_estimators=500, max_depth=4, min_samples_split=2,
			   learning_rate=0.01, show_graph=False):
	print("run:", run_name)


	# ===============================
	# Load data
	# ===============================
	boston = datasets.load_boston()
	X, y = shuffle(boston.data, boston.target, random_state=13)
	X = X.astype(np.float32)
	offset = int(X.shape[0] * 0.9)
	X_train, y_train = X[:offset], y[:offset]
	X_test, y_test = X[offset:], y[offset:]

	with mlflow.start_run(run_name=run_name) as run:
		mlflow.log_param("MLflow version", mlflow.version.VERSION)

		params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
				  'learning_rate': learning_rate, 'loss': 'ls'}

		#mlflow.log_params(params)

		clf = ensemble.GradientBoostingRegressor(**params)

		clf.fit(X_train, y_train)

		mlflow.sklearn.log_model(clf, "GradientBoostingRegressor")

		y_pred = clf.predict(X_test)

		# calculate error metrics
		mae = metrics.mean_absolute_error(y_test, y_pred)
		mse = metrics.mean_squared_error(y_test, y_pred)
		rsme = np.sqrt(mse)
		r2 = metrics.r2_score(y_test, y_pred)

		# Log metrics
		mlflow.log_metric("mae", mae)
		mlflow.log_metric("mse", mse)
		mlflow.log_metric("rsme", rsme)
		mlflow.log_metric("r2", r2)

		# #############################################################################
		# Plot training deviance

		# compute test set deviance
		test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

		for i, y_pred in enumerate(clf.staged_predict(X_test)):
			test_score[i] = clf.loss_(y_test, y_pred)

		plt.figure(figsize=(12, 6))
		plt.subplot(1, 2, 1)
		plt.title('Deviance')
		plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
				 label='Training Set Deviance')
		plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
				 label='Test Set Deviance')
		plt.legend(loc='upper right')
		plt.xlabel('Boosting Iterations')
		plt.ylabel('Deviance')

		# #############################################################################
		# Plot feature importance
		feature_importance = clf.feature_importances_

		# log the feature importance
		fi_pair_set = set(zip(boston.feature_names, feature_importance))
		for fi_pair in fi_pair_set:
			mlflow.log_metric("fi_" + fi_pair[0], round(fi_pair[1], 2))

		# make importances relative to max importance
		feature_importance = 100.0 * (feature_importance / feature_importance.max())
		sorted_idx = np.argsort(feature_importance)
		pos = np.arange(sorted_idx.shape[0]) + .5
		plt.subplot(1, 2, 2)
		plt.barh(pos, feature_importance[sorted_idx], align='center')
		plt.yticks(pos, boston.feature_names[sorted_idx])
		plt.xlabel('Relative Importance')
		plt.title('Variable Importance')

		plt.savefig("/tmp/deviance_feature_importance.png")
		mlflow.log_artifact("/tmp/deviance_feature_importance.png")

		if (show_graph):
			plt.show()

		# get current run and experiment id
		runID = run.info.run_uuid
		experimentID = run.info.experiment_id

		return (experimentID, runID)

if __name__ == "__main__":
	warnings.filterwarnings("ignore")

	print("mlflow version: " , mlflow.version.VERSION)

	mlflow_run()


