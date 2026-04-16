clc;
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'email_classification_data.csv';

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(3);
    n = 150;

    Promo_keywords = randi([0 5], n, 1);
    Suspicious_words = randi([0 1], n, 1);   % 0=No, 1=Yes
    Message_length = randi([40 250], n, 1);
    URL_count = randi([0 4], n, 1);

    Category = strings(n,1);

    for i = 1:n
        if Promo_keywords(i) >= 3 && Suspicious_words(i)==1 && URL_count(i)>=2
            Category(i) = "Spam";
        elseif Promo_keywords(i) >= 2 && Message_length(i) > 120
            Category(i) = "Promotional";
        else
            Category(i) = "Important";
        end
    end

    data = table(Promo_keywords, Suspicious_words, Message_length, URL_count, Category);
else
    data = readtable(csvFile);
    % Expected columns:
    % Promo_keywords, Suspicious_words, Message_length, URL_count, Category
end

disp(head(data));

%% =========================
% PREPROCESSING
% ==========================
data = rmmissing(data);
data = unique(data);

% Convert target to categorical
data.Category = categorical(data.Category);

% If Suspicious_words is text Yes/No
if iscellstr(data.Suspicious_words) || isstring(data.Suspicious_words) || iscategorical(data.Suspicious_words)
    data.Suspicious_words = grp2idx(categorical(data.Suspicious_words)) - 1;
end

X = [data.Promo_keywords, data.Suspicious_words, data.Message_length, data.URL_count];
Y = data.Category;

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
model = fitcnb(X_train, Y_train);

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

%% =========================
% BEST PREDICTED CLASS
% ==========================
classAcc = diag(confMat) ./ (sum(confMat,2) + eps);
[bestVal, bestIdx] = max(classAcc);
fprintf('\nBest predicted class: %s (%.4f)\n', string(classes(bestIdx)), bestVal);

%% =========================
% VISUALIZATION
% ==========================
figure;
confusionchart(Y_test, Y_pred);
title('Naive Bayes Confusion Matrix');
