clc;  % AdaBoost
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'adaboost_driver_risk.csv';
numLearners = 50;

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(21);
    n = 180;

    Avg_Speed = randi([60 140], n, 1);
    Harsh_Brakes = randi([0 20], n, 1);
    Night_Driving_Hours = randi([0 25], n, 1);
    Mobile_Usage = randi([0 60], n, 1);
    Accident_History = randi([0 4], n, 1);

    riskScore = 0.03*Avg_Speed + 0.18*Harsh_Brakes + 0.12*Night_Driving_Hours ...
                + 0.08*Mobile_Usage + 0.9*Accident_History;

    Risk_Level = double(riskScore > median(riskScore));

    data = table(Avg_Speed, Harsh_Brakes, Night_Driving_Hours, ...
                 Mobile_Usage, Accident_History, Risk_Level);
else
    data = readtable(csvFile);
end

disp(head(data));

%% PREPROCESSING
data = rmmissing(data);
data = unique(data);

X = [data.Avg_Speed, data.Harsh_Brakes, data.Night_Driving_Hours, ...
     data.Mobile_Usage, data.Accident_History];
Y = data.Risk_Level;

%% TRAIN-TEST SPLIT
cv = cvpartition(Y, 'HoldOut', 0.2);
X_train = X(training(cv), :);
Y_train = Y(training(cv), :);
X_test  = X(test(cv), :);
Y_test  = Y(test(cv), :);

%% MODEL BUILDING
model = fitcensemble(X_train, Y_train, ...
    'Method', 'AdaBoostM1', ...
    'NumLearningCycles', numLearners);

%% PREDICTION
Y_pred = predict(model, X_test);

%% EVALUATION
confMat = confusionmat(Y_test, Y_pred);
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
confusionchart(Y_test, Y_pred);
title('AdaBoost Confusion Matrix');
