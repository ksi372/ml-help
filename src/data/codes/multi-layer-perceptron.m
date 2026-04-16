clc;  % multi-layer perceptron (patternnet)
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'mlp_disease_risk.csv';

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(24);
    n = 180;

    Age_Group = randi([0 2], n, 1);         % Young=0, Middle=1, Senior=2
    BMI_Category = randi([0 2], n, 1);      % Normal=0, Overweight=1, Obese=2
    Physical_Activity = randi([0 2], n, 1); % Low=0, Moderate=1, High=2
    Blood_Pressure = randi([0 2], n, 1);    % Normal=0, Elevated=1, High=2
    Family_History = randi([0 1], n, 1);    % No=0, Yes=1

    riskScore = 0.7*Age_Group + 1.0*BMI_Category - 0.8*Physical_Activity ...
                + 0.9*Blood_Pressure + 1.2*Family_History;
    Risk_Level = double(riskScore >= 2.5);

    data = table(Age_Group, BMI_Category, Physical_Activity, ...
                 Blood_Pressure, Family_History, Risk_Level);
else
    data = readtable(csvFile);
end

disp(head(data));

%% PREPROCESSING
data = rmmissing(data);
data = unique(data);

X = [data.Age_Group, data.BMI_Category, data.Physical_Activity, ...
     data.Blood_Pressure, data.Family_History]';
Y = data.Risk_Level';

%% TRAIN-TEST SPLIT
cv = cvpartition(Y, 'HoldOut', 0.2);
X_train = X(:, training(cv));
Y_train = Y(:, training(cv));
X_test  = X(:, test(cv));
Y_test  = Y(:, test(cv));

%% MODEL BUILDING
net = patternnet(10);   % one hidden layer with 10 neurons
net = train(net, X_train, Y_train);

%% PREDICTION
Y_prob = net(X_test);
Y_pred = round(Y_prob);

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
title('MLP Confusion Matrix');
