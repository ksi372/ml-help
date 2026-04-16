clc;  % KNN
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'knn_student_data.csv';
K = 5;   % change this to study effect of K

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(10);
    n = 150;

    Study_Hours = randi([1 10], n, 1);
    Attendance = randi([50 100], n, 1);
    Assignment_Score = randi([40 100], n, 1);
    Internal_Marks = randi([10 50], n, 1);

    score = 0.8*Study_Hours + 0.03*Attendance + 0.03*Assignment_Score + 0.08*Internal_Marks;
    Result = double(score > 8.5);

    data = table(Study_Hours, Attendance, Assignment_Score, Internal_Marks, Result);
else
    data = readtable(csvFile);
    % Expected columns:
    % Study_Hours, Attendance, Assignment_Score, Internal_Marks, Result
end

disp(head(data));

%% =========================
% PREPROCESSING
% ==========================
data = rmmissing(data);
data = unique(data);

X = [data.Study_Hours, data.Attendance, data.Assignment_Score, data.Internal_Marks];
Y = data.Result;

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
model = fitcknn(X_train, Y_train, 'NumNeighbors', K);

%% =========================
% PREDICTION
% ==========================
Y_pred = predict(model, X_test);

%% =========================
% EVALUATION
% ==========================
confMat = confusionmat(Y_test, Y_pred);
accuracy = sum(diag(confMat)) / sum(confMat(:));

TP = confMat(2,2); TN = confMat(1,1);
FP = confMat(1,2); FN = confMat(2,1);

precision = TP / (TP + FP + eps);
recall    = TP / (TP + FN + eps);
f1        = 2 * precision * recall / (precision + recall + eps);

disp('Confusion Matrix:');
disp(confMat);
fprintf('\nK = %d\n', K);
fprintf('Accuracy  = %.4f\n', accuracy);
fprintf('Precision = %.4f\n', precision);
fprintf('Recall    = %.4f\n', recall);
fprintf('F1-score  = %.4f\n', f1);

%% =========================
% EFFECT OF K VALUE
% ==========================
k_values = 1:10;
acc_values = zeros(size(k_values));

for i = 1:length(k_values)
    tempModel = fitcknn(X_train, Y_train, 'NumNeighbors', k_values(i));
    tempPred = predict(tempModel, X_test);
    tempConf = confusionmat(Y_test, tempPred);
    acc_values(i) = sum(diag(tempConf)) / sum(tempConf(:));
end

figure;
plot(k_values, acc_values, 'o-', 'LineWidth', 1.5);
xlabel('K value');
ylabel('Accuracy');
title('Effect of K on KNN Accuracy');
grid on;

figure;
confusionchart(Y_test, Y_pred);
title('KNN Confusion Matrix');
