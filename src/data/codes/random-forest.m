clc;  % random forest
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'random_forest_loan.csv';
numTrees = 100;

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(15);
    n = 180;

    Annual_Income = 2 + 8*rand(n,1);           % lakhs
    Credit_Score = randi([450 850], n, 1);
    Employment_Years = randi([0 15], n, 1);
    Loan_Amount = 2 + 10*rand(n,1);
    Existing_Loans = randi([0 3], n, 1);

    score = 0.7*Annual_Income + 0.01*Credit_Score + 0.15*Employment_Years ...
            - 0.5*Loan_Amount - 0.8*Existing_Loans;

    Loan_Status = double(score > 7.5);

    data = table(Annual_Income, Credit_Score, Employment_Years, Loan_Amount, Existing_Loans, Loan_Status);
else
    data = readtable(csvFile);
    % Expected columns:
    % Annual_Income, Credit_Score, Employment_Years, Loan_Amount, Existing_Loans, Loan_Status
end

disp(head(data));

%% =========================
% PREPROCESSING
% ==========================
data = rmmissing(data);
data = unique(data);

X = [data.Annual_Income, data.Credit_Score, data.Employment_Years, data.Loan_Amount, data.Existing_Loans];
Y = data.Loan_Status;

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
model = TreeBagger(numTrees, X_train, Y_train, ...
    'Method', 'classification', ...
    'OOBPrediction', 'On');

%% =========================
% PREDICTION
% ==========================
[Y_pred_cell, ~] = predict(model, X_test);
Y_pred = str2double(Y_pred_cell);

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
fprintf('\nNumber of Trees = %d\n', numTrees);
fprintf('Accuracy  = %.4f\n', accuracy);
fprintf('Precision = %.4f\n', precision);
fprintf('Recall    = %.4f\n', recall);
fprintf('F1-score  = %.4f\n', f1);

%% =========================
% EFFECT OF NUMBER OF TREES
% ==========================
treeVals = [10 20 50 100 150];
accVals = zeros(size(treeVals));

for i = 1:length(treeVals)
    tempModel = TreeBagger(treeVals(i), X_train, Y_train, 'Method', 'classification');
    tempPredCell = predict(tempModel, X_test);
    tempPred = str2double(tempPredCell);
    tempConf = confusionmat(Y_test, tempPred);
    accVals(i) = sum(diag(tempConf)) / sum(tempConf(:));
end

figure;
plot(treeVals, accVals, 'o-', 'LineWidth', 1.5);
xlabel('Number of Trees');
ylabel('Accuracy');
title('Effect of Number of Trees on Random Forest Accuracy');
grid on;

figure;
confusionchart(Y_test, Y_pred);
title('Random Forest Confusion Matrix');
