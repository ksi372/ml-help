import type { Topic } from '../types'
import linearRegression from './codes/linear-regression.m?raw'
import qLearning from './codes/q-learning.m?raw'
import naiveBayes from './codes/naive-bayes.m?raw'
import decisionTree from './codes/decision-tree.m?raw'
import cart from './codes/cart.m?raw'
import logisticRegression from './codes/logistic-regression.m?raw'
import knn from './codes/knn.m?raw'
import linearSvm from './codes/linear-svm.m?raw'
import nonlinearSvmRbf from './codes/nonlinear-svm-rbf.m?raw'
import kmeans from './codes/kmeans.m?raw'
import kmodes from './codes/kmodes.m?raw'
import randomForest from './codes/random-forest.m?raw'
import adaboost from './codes/adaboost.m?raw'
import xgboostLsboost from './codes/xgboost-lsboost.m?raw'
import singleLayerPerceptron from './codes/single-layer-perceptron.m?raw'
import multiLayerPerceptron from './codes/multi-layer-perceptron.m?raw'
import pca from './codes/pca.m?raw'
import som from './codes/som.m?raw'

/** All snippets ship with the app (offline). */
export const TOPICS: Topic[] = [
  {
    id: 'linear-regression',
    title: 'Linear Regression',
    hint: 'fitlm · normalize · MAE / MSE / RMSE / R²',
    code: linearRegression.trimEnd(),
  },
  {
    id: 'q-learning',
    title: 'Q-Learning',
    hint: 'ε-greedy · grid world · Q-update · policy grid',
    code: qLearning.trimEnd(),
  },
  {
    id: 'naive-bayes',
    title: 'Naive Bayes',
    hint: 'fitcnb · confusion matrix · per-class precision / recall / F1',
    code: naiveBayes.trimEnd(),
  },
  {
    id: 'decision-tree',
    title: 'Decision Tree (ID3-style)',
    hint: "fitctree · SplitCriterion 'deviance'",
    code: decisionTree.trimEnd(),
  },
  {
    id: 'cart',
    title: 'CART',
    hint: "fitctree · SplitCriterion 'gdi' (Gini)",
    code: cart.trimEnd(),
  },
  {
    id: 'logistic-regression',
    title: 'Logistic Regression',
    hint: 'fitclinear · logistic · ridge · probabilities',
    code: logisticRegression.trimEnd(),
  },
  {
    id: 'knn',
    title: 'KNN (k-Nearest Neighbors)',
    hint: 'fitcknn · NumNeighbors · sweep K vs accuracy',
    code: knn.trimEnd(),
  },
  {
    id: 'linear-svm',
    title: 'Linear SVM',
    hint: 'fitcsvm · KernelFunction linear',
    code: linearSvm.trimEnd(),
  },
  {
    id: 'nonlinear-svm-rbf',
    title: 'Nonlinear SVM (RBF)',
    hint: 'fitcsvm · KernelFunction rbf',
    code: nonlinearSvmRbf.trimEnd(),
  },
  {
    id: 'kmeans',
    title: 'K-Means',
    hint: 'kmeans · inertia · silhouette · gscatter',
    code: kmeans.trimEnd(),
  },
  {
    id: 'kmodes',
    title: 'K-Modes (categorical)',
    hint: 'modes · matching distance · cluster cost',
    code: kmodes.trimEnd(),
  },
  {
    id: 'random-forest',
    title: 'Random Forest',
    hint: 'TreeBagger · OOB · tree count sweep',
    code: randomForest.trimEnd(),
  },
  {
    id: 'adaboost',
    title: 'AdaBoost',
    hint: 'fitcensemble · AdaBoostM1 · NumLearningCycles',
    code: adaboost.trimEnd(),
  },
  {
    id: 'xgboost-lsboost',
    title: 'Gradient boosting (LSBoost)',
    hint: 'fitrensemble · LSBoost + templateTree (MATLAB, not Python XGBoost)',
    code: xgboostLsboost.trimEnd(),
  },
  {
    id: 'single-layer-perceptron',
    title: 'Single-layer perceptron',
    hint: 'perceptron · train · column patterns',
    code: singleLayerPerceptron.trimEnd(),
  },
  {
    id: 'multi-layer-perceptron',
    title: 'Multi-layer perceptron (MLP)',
    hint: 'patternnet · hidden neurons · confusionchart',
    code: multiLayerPerceptron.trimEnd(),
  },
  {
    id: 'pca',
    title: 'PCA',
    hint: 'pca · explained variance · biplot-style scatter',
    code: pca.trimEnd(),
  },
  {
    id: 'som',
    title: 'SOM (self-organizing map)',
    hint: 'selforgmap · plotsomtop / hits / nd / pos',
    code: som.trimEnd(),
  },
  {
    id: 'ui-spacing-notes',
    title: 'Spacing & layout',
    hint: 'margins · padding · readability',
    kind: 'stealth-gemini',
  },
]
