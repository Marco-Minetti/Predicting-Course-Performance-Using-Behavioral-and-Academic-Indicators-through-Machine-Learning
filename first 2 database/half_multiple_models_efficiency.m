clear; clc; close all;

%% Load dataset
T = readtable("Student_Performance_with_efficiency.csv");

% Convert categorical columns
catVars = ["Gender","Class","Parental_Education","Internet_Access","Extracurricular_Activities"];
for v = catVars
    if ismember(v, T.Properties.VariableNames)
        T.(v) = categorical(T.(v));
    end
end

% Remove ID if present
if ismember("Student_ID", T.Properties.VariableNames)
    T.Student_ID = [];
end

% Remove rows with missing values
T = rmmissing(T);

%% Target
target = "Final_Percentage";

%% Choose predictor setting
% true  = include Math/Science/English/Previous_Year_Score
% false = remove them for a harder, more realistic test
useStrongAcademicPredictors = true;
useStrongEfficencyPredictors = false;

if ~useStrongAcademicPredictors
    varsToRemove = ["Math_Score","Science_Score","English_Score","Previous_Year_Score","Performance_Level"];
    for v = varsToRemove
        if ismember(v, T.Properties.VariableNames)
            T.(v) = [];
        end
    end
end

if ~useStrongEfficencyPredictors
    varsToRemove = ["Attendance_Percentage","Study_Hours_Per_Day", "Efficiency_Level"];
    for v = varsToRemove
        if ismember(v, T.Properties.VariableNames)
            T.(v) = [];
        end
    end
end

%% Half split
n = height(T);
splitIndex = floor(n/2);

trainT = T(1:splitIndex, :);
testT  = T(splitIndex+1:end, :);

yTrain = trainT.(target);
yTest  = testT.(target);

XTrain = removevars(trainT, target);
XTest  = removevars(testT, target);

fprintf("Training rows: %d\n", height(trainT));
fprintf("Testing rows : %d\n\n", height(testT));

%% Create folder for figures
figFolder = "model_figures";
if ~exist(figFolder, 'dir')
    mkdir(figFolder);
end

%% Train many regression models
models = {};
modelNames = {};

% 1. Linear Regression
try
    models{end+1} = fitrlinear(XTrain, yTrain);
    modelNames{end+1} = "Linear Regression";
catch ME
    fprintf("Skipping Linear Regression: %s\n", ME.message);
end

% 2. Regression Tree
try
    models{end+1} = fitrtree(XTrain, yTrain);
    modelNames{end+1} = "Regression Tree";
catch ME
    fprintf("Skipping Regression Tree: %s\n", ME.message);
end

% 3. Bagged Trees
try
    models{end+1} = fitrensemble(XTrain, yTrain, ...
        "Method", "Bag", ...
        "NumLearningCycles", 200);
    modelNames{end+1} = "Bagged Trees";
catch ME
    fprintf("Skipping Bagged Trees: %s\n", ME.message);
end

% 4. LSBoost Ensemble
try
    models{end+1} = fitrensemble(XTrain, yTrain, ...
        "Method", "LSBoost", ...
        "NumLearningCycles", 300);
    modelNames{end+1} = "LSBoost Ensemble";
catch ME
    fprintf("Skipping LSBoost Ensemble: %s\n", ME.message);
end

% 5. Gaussian Process Regression
try
    models{end+1} = fitrgp(XTrain, yTrain);
    modelNames{end+1} = "Gaussian Process Regression";
catch ME
    fprintf("Skipping Gaussian Process Regression: %s\n", ME.message);
end

% 6. Support Vector Regression
try
    models{end+1} = fitrsvm(XTrain, yTrain, ...
        "KernelFunction", "gaussian", ...
        "Standardize", true);
    modelNames{end+1} = "Support Vector Regression";
catch ME
    fprintf("Skipping Support Vector Regression: %s\n", ME.message);
end

% 7. KNN Regression
try
    models{end+1} = fitrknn(XTrain, yTrain, ...
        "NumNeighbors", 10, ...
        "Standardize", true);
    modelNames{end+1} = "KNN Regression";
catch ME
    fprintf("Skipping KNN Regression: %s\n", ME.message);
end

%% Check that at least one model was trained
numModels = numel(models);
if numModels == 0
    error("No models were successfully trained.");
end

%% Results table
Results = table('Size', [numModels 4], ...
    'VariableTypes', ["string","double","double","double"], ...
    'VariableNames', ["Model","RMSE","MAE","R2"]);

fprintf("===== MODEL RESULTS =====\n");

%% Evaluate all models and create figures
for i = 1:numModels

    yPred = predict(models{i}, XTest);

    % Metrics
    rmse = sqrt(mean((yPred - yTest).^2));
    mae  = mean(abs(yPred - yTest));
    r2   = 1 - sum((yPred - yTest).^2) / sum((yTest - mean(yTest)).^2);

    Results.Model(i) = string(modelNames{i});
    Results.RMSE(i) = rmse;
    Results.MAE(i) = mae;
    Results.R2(i) = r2;

    fprintf("%s | RMSE: %.4f | MAE: %.4f | R^2: %.6f\n", ...
        modelNames{i}, rmse, mae, r2);

    % Safe name for files
    safeName = strrep(modelNames{i}, " ", "_");

    %% Figure 1: Predicted vs Actual
    minVal = min([yTest; yPred]);
    maxVal = max([yTest; yPred]);

    f1 = figure('Visible','on');
    scatter(yTest, yPred, 18, 'filled');
    hold on;
    plot([minVal maxVal], [minVal maxVal], 'r--', 'LineWidth', 1.5);
    grid on;
    xlabel("Actual Final Percentage");
    ylabel("Predicted Final Percentage");
    title(modelNames{i} + " - Predicted vs Actual (R^2 = " + num2str(r2, "%.3f") + ")");
    hold off;

    saveas(f1, fullfile(figFolder, safeName + "_pred_vs_actual.png"));

    %% Figure 2: Residual Plot
    residuals = yTest - yPred;

    f2 = figure('Visible','on');
    scatter(yPred, residuals, 18, 'filled');
    hold on;
    yline(0, 'r--', 'LineWidth', 1.5);
    grid on;
    xlabel("Predicted Final Percentage");
    ylabel("Residuals");
    title(modelNames{i} + " - Residual Plot");
    hold off;

    saveas(f2, fullfile(figFolder, safeName + "_residuals.png"));

end

%% Sort results by RMSE
Results = sortrows(Results, "RMSE");

disp(" ");
disp("===== SORTED RESULTS (best first) =====");
disp(Results);

%% Save results table
writetable(Results, "regression_model_results.csv");
fprintf("\nResults saved to regression_model_results.csv\n");
fprintf("Figures saved in folder: %s\n", figFolder);

%% Show best model again
bestModelName = Results.Model(1);
fprintf("\nBest model: %s\n", bestModelName);