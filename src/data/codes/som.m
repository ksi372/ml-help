clc;  % SOM (self-organizing map)
clear;
close all;

%% =========================
% SETTINGS
% ==========================
useSynthetic = true;
csvFile = 'som_health_data.csv';

%% =========================
% LOAD OR CREATE DATA
% ==========================
if useSynthetic
    rng(26);
    n = 180;

    Age = randi([20 70], n, 1);
    BMI = 18 + 18*rand(n,1);
    Blood_Pressure = randi([60 100], n, 1);
    Glucose = randi([80 190], n, 1);
    Cholesterol = randi([150 280], n, 1);

    risk = 0.03*Age + 0.08*BMI + 0.03*Blood_Pressure + 0.03*Glucose + 0.01*Cholesterol;
    Outcome = double(risk > median(risk));

    Patient_ID = strcat("P", string((1:n)'));
    data = table(Patient_ID, Age, BMI, Blood_Pressure, Glucose, Cholesterol, Outcome);
else
    data = readtable(csvFile);
end

disp(head(data));

%% PREPROCESSING
data = rmmissing(data);
data = unique(data);

X = [data.Age, data.BMI, data.Blood_Pressure, data.Glucose, data.Cholesterol]';
X = normalize(X, 2);

%% MODEL BUILDING
net = selforgmap([10 10]);  % 10x10 grid
net = train(net, X);

%% MAP RESPONSES
y = net(X);

%% VISUALIZATION
figure;
plotsomtop(net);
title('SOM Topology');

figure;
plotsomhits(net, X);
title('SOM Hits Map');

figure;
plotsomnd(net);
title('SOM Neighbor Distances');

figure;
plotsompos(net, X);
title('SOM Position Map');
