clc;
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'fraud_detection_data.csv';

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(5);
    n = 180;

    Transaction_amount = randi([500 80000], n, 1);
    Transactions_per_day = randi([1 8], n, 1);
    Location_mismatch = randi([0 1], n, 1);   % 0=No,1=Yes
    Device_change = randi([0 1], n, 1);       % 0=No,1=Yes
    Transaction_time = randi([0 1], n, 1);    % 0=Day,1=Night

    Fraud_Status = strings(n,1);

    for i = 1:n
        risk = 0;
        if Transaction_amount(i) > 30000, risk = risk + 1; end
        if Transactions_per_day(i) > 4, risk = risk + 1; end
        if Location_mismatch(i) == 1, risk = risk + 1; end
        if Device_change(i) == 1, risk = risk + 1; end
        if Transaction_time(i) == 1, risk = risk + 1; end

        if risk >= 3
            Fraud_Status(i) = "Fraud";
        else
            Fraud_Status(i) = "Legitimate";
        end
    end

    data = table(Transaction_amount, Transactions_per_day, Location_mismatch, ...
                 Device_change, Transaction_time, Fraud_Status);
else
    data = readtable(csvFile);
    % Expected columns:
    % Transaction_amount, Transactions_per_day, Location_mismatch,
    % Device_change, Transaction_time, Fraud_Status
end

disp(head(data));

%% =========================
% PREPROCESSING
% ==========================
data = rmmissing(data);
data = unique(data);

vars = {'Location_mismatch','Device_change','Transaction_time'};
for i = 1:length(vars)
    if iscellstr(data.(vars{i})) || isstring(data.(vars{i})) || iscategorical(data.(vars{i}))
        data.(vars{i}) = grp2idx(categorical(data.(vars{i}))) - 1;
    end
end

data.Fraud_Status = categorical(data.Fraud_Status);

X = [data.Transaction_amount, data.Transactions_per_day, data.Location_mismatch, ...
     data.Device_change, data.Transaction_time];
Y = data.Fraud_Status;

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
model = fitctree(X_train, Y_train, 'SplitCriterion', 'gdi');

%% =========================
% VIEW TREE
% ==========================
view(model, 'Mode', 'graph');

%% =========================
% PREDICTION
% ==========================
Y_pred = predict(model, X_test);

%% =========================
% EVALUATION
% ==========================
confMat = confusionmat(Y_test, Y_pred);
accuracy = sum(diag(confMat)) / sum(confMat(:));

disp('Confusion Matrix:');
disp(confMat);
fprintf('\nAccuracy = %.4f\n', accuracy);

classes = categories(Y);
for i = 1:length(classes)
    TP = confMat(i,i);
    FP = sum(confMat(:,i)) - TP;
    FN = sum(confMat(i,:)) - TP;

    precision = TP / (TP + FP + eps);
    recall    = TP / (TP + FN + eps);
    f1        = 2 * precision * recall / (precision + recall + eps);

    fprintf('\nClass: %s\n', string(classes(i)));
    fprintf('Precision = %.4f\n', precision);
    fprintf('Recall    = %.4f\n', recall);
    fprintf('F1-score  = %.4f\n', f1);
end

figure;
confusionchart(Y_test, Y_pred);
title('CART Confusion Matrix');
