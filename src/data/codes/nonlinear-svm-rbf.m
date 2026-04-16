clc;
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'rbf_svm_tumor.csv';

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(12);
    n = 160;

    Radius = 10 + 8*rand(n,1);
    Texture = 12 + 12*rand(n,1);
    Smoothness = 0.07 + 0.12*rand(n,1);
    Symmetry = 0.10 + 0.18*rand(n,1);

    % Nonlinear-like relation
    risk = (Radius - 14).^2 + (Texture - 18).^2 + 60*Smoothness + 40*Symmetry;
    Diagnosis = double(risk > median(risk));

    data = table(Radius, Texture, Smoothness, Symmetry, Diagnosis);
else
    data = readtable(csvFile);
    % Expected columns:
    % Radius, Texture, Smoothness, Symmetry, Diagnosis
end

disp(head(data));

%% =========================
% PREPROCESSING
% ==========================
data = rmmissing(data);
data = unique(data);

X = [data.Radius, data.Texture, data.Smoothness, data.Symmetry];
Y = data.Diagnosis;

X = normalize(X);

%% =========================
% TRAIN-TEST SPLIT
% ==========================
cv = cvpartition(Y, 'HoldOut', 0.2);
X_train = X(training(cv), :);
Y_train = Y(training(cv), :);
X_test  = X(test(cv), :);
Y_test  = Y(test(cv), :);

%% =========================
% MODEL BUILDING
% ==========================
model = fitcsvm(X_train, Y_train, ...
    'KernelFunction', 'rbf', ...
    'Standardize', true);

%% =========================
% PREDICTION
% ==========================
Y_pred = predict(model, X_test);

%% =========================
% EVALUATION
% ==========================
confMat = confusionmat(Y_test, Y_pred);
accuracy = sum(diag(confMat)) / sum(confMat(:));

TP = confMat(2,2); FP = confMat(1,2); FN = confMat(2,1);
precision = TP / (TP + FP + eps);
recall    = TP / (TP + FN + eps);
f1        = 2 * precision * recall / (precision + recall + eps);

disp('Confusion Matrix:');
disp(confMat);
fprintf('\nAccuracy  = %.4f\n', accuracy);
fprintf('Precision = %.4f\n', precision);
fprintf('Recall    = %.4f\n', recall);
fprintf('F1-score  = %.4f\n', f1);

figure;
confusionchart(Y_test, Y_pred);
title('RBF SVM Confusion Matrix');
