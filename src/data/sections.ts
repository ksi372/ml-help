import type { Topic } from '../types'
import linearRegression from './codes/linear-regression.py?raw'
import qLearning from './codes/q-learning.py?raw'
import naiveBayes from './codes/naive-bayes.py?raw'
import decisionTree from './codes/decision-tree.py?raw'
import cart from './codes/cart.py?raw'
import logisticRegression from './codes/logistic-regression.py?raw'
import knn from './codes/knn.py?raw'
import linearSvm from './codes/linear-svm.py?raw'
import nonlinearSvmRbf from './codes/nonlinear-svm-rbf.py?raw'
import kmeans from './codes/kmeans.py?raw'
import kmodes from './codes/kmodes.py?raw'
import randomForest from './codes/random-forest.py?raw'
import adaboost from './codes/adaboost.py?raw'
import xgboostLsboost from './codes/xgboost-lsboost.py?raw'
import singleLayerPerceptron from './codes/single-layer-perceptron.py?raw'
import multiLayerPerceptron from './codes/multi-layer-perceptron.py?raw'
import pca from './codes/pca.py?raw'
import som from './codes/som.py?raw'

/** All snippets ship with the app (offline). */
export const TOPICS: Topic[] = [
  {
    id: 'linear-regression',
    title: 'Linear Regression',
    hint: 'sklearn LinearRegression · metrics · plots',
    code: linearRegression.trimEnd(),
  },
  {
    id: 'q-learning',
    title: 'Q-Learning',
    hint: 'ε-greedy · tabular Q · grid world',
    code: qLearning.trimEnd(),
  },
  {
    id: 'naive-bayes',
    title: 'Naive Bayes',
    hint: 'GaussianNB · confusion matrix · classification_report',
    code: naiveBayes.trimEnd(),
  },
  {
    id: 'decision-tree',
    title: 'Decision Tree (log loss)',
    hint: 'DecisionTreeClassifier criterion=log_loss',
    code: decisionTree.trimEnd(),
  },
  {
    id: 'cart',
    title: 'CART',
    hint: 'DecisionTreeClassifier criterion=gini',
    code: cart.trimEnd(),
  },
  {
    id: 'logistic-regression',
    title: 'Logistic Regression',
    hint: 'LogisticRegression L2 · predict_proba',
    code: logisticRegression.trimEnd(),
  },
  {
    id: 'knn',
    title: 'KNN (k-Nearest Neighbors)',
    hint: 'KNeighborsClassifier · sweep K',
    code: knn.trimEnd(),
  },
  {
    id: 'linear-svm',
    title: 'Linear SVM',
    hint: 'SVC kernel=linear',
    code: linearSvm.trimEnd(),
  },
  {
    id: 'nonlinear-svm-rbf',
    title: 'Nonlinear SVM (RBF)',
    hint: 'SVC kernel=rbf',
    code: nonlinearSvmRbf.trimEnd(),
  },
  {
    id: 'kmeans',
    title: 'K-Means',
    hint: 'KMeans · inertia · silhouette',
    code: kmeans.trimEnd(),
  },
  {
    id: 'kmodes',
    title: 'K-Modes (categorical)',
    hint: 'matching distance · modes',
    code: kmodes.trimEnd(),
  },
  {
    id: 'random-forest',
    title: 'Random Forest',
    hint: 'RandomForestClassifier · n_estimators sweep',
    code: randomForest.trimEnd(),
  },
  {
    id: 'adaboost',
    title: 'AdaBoost',
    hint: 'AdaBoostClassifier + stump',
    code: adaboost.trimEnd(),
  },
  {
    id: 'xgboost-lsboost',
    title: 'Gradient boosting (regression)',
    hint: 'GradientBoostingRegressor (sklearn boosting pipeline)',
    code: xgboostLsboost.trimEnd(),
  },
  {
    id: 'single-layer-perceptron',
    title: 'Single-layer perceptron',
    hint: 'sklearn Perceptron',
    code: singleLayerPerceptron.trimEnd(),
  },
  {
    id: 'multi-layer-perceptron',
    title: 'Multi-layer perceptron (MLP)',
    hint: 'MLPClassifier hidden_layer_sizes',
    code: multiLayerPerceptron.trimEnd(),
  },
  {
    id: 'pca',
    title: 'PCA',
    hint: 'PCA · explained_variance_ratio_ · scatter PC1–PC2',
    code: pca.trimEnd(),
  },
  {
    id: 'som',
    title: 'SOM (self-organizing map)',
    hint: 'MiniSom (pip install minisom) · U-matrix',
    code: som.trimEnd(),
  },
  {
    id: 'ui-spacing-notes',
    title: 'Spacing & layout',
    hint: 'margins · padding · readability',
    kind: 'stealth-gemini',
  },
]
