clc;
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'linear_svm_placement.csv';

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(11);
    n = 140;

    CGPA = 5 + 5*rand(n,1);
    Aptitude_Score = randi([40 100], n, 1);
    Coding_Score = randi([35 100], n, 1);
    Internship = randi([0 1], n, 1);

    score = 1.2*CGPA + 0.03*Aptitude_Score + 0.035*Coding_Score + 0.8*Internship;
    Placement = double(score > 11);

    data = table(CGPA, Aptitude_Score, Coding_Score, Internship, Placement);
else
    data = readtable(csvFile);
    % Expected columns:
    % CGPA, Aptitude_Score, Coding_Score, Internship, Placement
end

disp(head(data));

%% =========================
% PREPROCESSING
% ==========================
data = rmmissing(data);
data = unique(data);

X = [data.CGPA, data.Aptitude_Score, data.Coding_Score, data.Internship];
Y = data.Placement;

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
model = fitcsvm(X_train, Y_train, 'KernelFunction', 'linear', 'Standardize', true);

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
title('Linear SVM Confusion Matrix');
