clc;
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'diabetes_data.csv';

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(6);
    n = 160;

    Age = randi([20 70], n, 1);
    BMI = 18 + 18*rand(n,1);
    Glucose = randi([75 200], n, 1);
    Blood_Pressure = randi([60 100], n, 1);

    % Create probability using logistic-type relation
    z = -18 + 0.05*Age + 0.18*BMI + 0.04*Glucose + 0.03*Blood_Pressure;
    p = 1 ./ (1 + exp(-z));
    Diabetes = double(rand(n,1) < p);

    data = table(Age, BMI, Glucose, Blood_Pressure, Diabetes);
else
    data = readtable(csvFile);
    % Expected columns:
    % Age, BMI, Glucose, Blood_Pressure, Diabetes
end

disp(head(data));

%% =========================
% PREPROCESSING
% ==========================
data = rmmissing(data);
data = unique(data);

X = [data.Age, data.BMI, data.Glucose, data.Blood_Pressure];
Y = data.Diabetes;

% Normalize features
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
model = fitclinear(X_train, Y_train, ...
    'Learner', 'logistic', ...
    'Regularization', 'ridge');

%% =========================
% PREDICTION
% ==========================
[label, score] = predict(model, X_test);
Y_pred = label;              % 0 or 1
probDiabetes = score(:,2);   % probability for class 1

%% =========================
% EVALUATION
% ==========================
confMat = confusionmat(Y_test, Y_pred);
accuracy = sum(diag(confMat)) / sum(confMat(:));

TP = confMat(2,2);
TN = confMat(1,1);
FP = confMat(1,2);
FN = confMat(2,1);

precision = TP / (TP + FP + eps);
recall    = TP / (TP + FN + eps);
f1        = 2 * precision * recall / (precision + recall + eps);

disp('Confusion Matrix:');
disp(confMat);

fprintf('\nAccuracy  = %.4f\n', accuracy);
fprintf('Precision = %.4f\n', precision);
fprintf('Recall    = %.4f\n', recall);
fprintf('F1-score  = %.4f\n', f1);

%% =========================
% VISUALIZATION
% ==========================
figure;
confusionchart(Y_test, Y_pred);
title('Logistic Regression Confusion Matrix');

figure;
scatter(1:length(Y_test), probDiabetes, 'filled');
xlabel('Test Sample');
ylabel('Predicted Probability of Diabetes');
title('Predicted Probability');
grid on;
