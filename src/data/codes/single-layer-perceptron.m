clc;  % single-layer perceptron
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'slp_loan_approval.csv';

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(23);
    n = 160;

    Income_Level = randi([0 2], n, 1);       % Low=0, Medium=1, High=2
    Credit_Score = randi([0 2], n, 1);       % Low=0, Medium=1, High=2
    Employment_Status = randi([0 1], n, 1);  % Unstable=0, Stable=1
    Loan_Amount = randi([0 1], n, 1);        % Low=0, High=1

    score = 1.5*Income_Level + 1.5*Credit_Score + 1.2*Employment_Status - 1.0*Loan_Amount;
    Loan_Status = double(score >= 2.5);

    data = table(Income_Level, Credit_Score, Employment_Status, Loan_Amount, Loan_Status);
else
    data = readtable(csvFile);
end

disp(head(data));

%% PREPROCESSING
data = rmmissing(data);
data = unique(data);

X = [data.Income_Level, data.Credit_Score, data.Employment_Status, data.Loan_Amount]';
Y = data.Loan_Status';

%% TRAIN-TEST SPLIT
cv = cvpartition(Y, 'HoldOut', 0.2);
X_train = X(:, training(cv));
Y_train = Y(:, training(cv));
X_test  = X(:, test(cv));
Y_test  = Y(:, test(cv));

%% MODEL BUILDING
net = perceptron;
net = train(net, X_train, Y_train);

%% PREDICTION
Y_pred = net(X_test);
Y_pred = round(Y_pred);

%% EVALUATION
confMat = confusionmat(Y_test', Y_pred');
accuracy = sum(diag(confMat)) / sum(confMat(:));
TP = confMat(2,2); FP = confMat(1,2); FN = confMat(2,1);
precision = TP/(TP+FP+eps);
recall = TP/(TP+FN+eps);
f1 = 2*precision*recall/(precision+recall+eps);

disp('Confusion Matrix:');
disp(confMat);
fprintf('\nAccuracy  = %.4f\n', accuracy);
fprintf('Precision = %.4f\n', precision);
fprintf('Recall    = %.4f\n', recall);
fprintf('F1-score  = %.4f\n', f1);

figure;
confusionchart(Y_test', Y_pred');
title('Single Layer Perceptron Confusion Matrix');
