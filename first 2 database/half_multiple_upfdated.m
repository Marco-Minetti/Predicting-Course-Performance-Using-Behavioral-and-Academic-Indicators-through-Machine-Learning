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
useStrongAcademicPredictors = false;
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

%% Shuffled half split (fixed seed for reproducibility)
rng(42);
n = height(T);
shuffledIdx = randperm(n);
splitIndex = floor(n/2);

trainT = T(shuffledIdx(1:splitIndex), :);
testT  = T(shuffledIdx(splitIndex+1:end), :);

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

% 7. KNN Regression (corrected function)
try
    models{end+1} = fitcknn(XTrain, yTrain, ...
        "NumNeighbors", 10);
    modelNames{end+1} = "KNN Regression";
catch ME
    fprintf("Skipping KNN Regression: %s\n", ME.message);
end

%% Check that at least one model was trained
numModels = numel(models);
if numModels == 0
    error("No models were successfully trained.");
end

%% Results table — Train vs Test + Overfit Gap
Results = table('Size', [numModels 7], ...
    'VariableTypes', ["string","double","double","double","double","double","double"], ...
    'VariableNames', ["Model","Train_RMSE","Train_R2","Test_RMSE","Test_MAE","Test_R2","Overfit_Gap"]);

fprintf("===== MODEL RESULTS (Train vs Test) =====\n");

%% Evaluate all models and create figures
for i = 1:numModels

    % Predictions
    yPredTest  = predict(models{i}, XTest);
    yPredTrain = predict(models{i}, XTrain);

    % Test metrics
    test_rmse = sqrt(mean((yPredTest - yTest).^2));
    test_mae  = mean(abs(yPredTest - yTest));
    test_r2   = 1 - sum((yPredTest - yTest).^2) / sum((yTest - mean(yTest)).^2);

    % Train metrics
    train_rmse = sqrt(mean((yPredTrain - yTrain).^2));
    train_r2   = 1 - sum((yPredTrain - yTrain).^2) / sum((yTrain - mean(yTrain)).^2);

    % Overfit gap: how much better the model is on training vs test
    overfit_gap = train_r2 - test_r2;

    % Overfitting verdict
    if overfit_gap < 0.02
        verdict = "No overfitting";
    elseif overfit_gap < 0.05
        verdict = "Mild";
    elseif overfit_gap < 0.10
        verdict = "Moderate";
    else
        verdict = "Likely overfitting";
    end

    Results.Model(i)       = string(modelNames{i});
    Results.Train_RMSE(i)  = train_rmse;
    Results.Train_R2(i)    = train_r2;
    Results.Test_RMSE(i)   = test_rmse;
    Results.Test_MAE(i)    = test_mae;
    Results.Test_R2(i)     = test_r2;
    Results.Overfit_Gap(i) = overfit_gap;

    fprintf("%s\n  Train R²: %.4f | Test R²: %.4f | Gap: %.4f => %s\n  Test RMSE: %.4f | Test MAE: %.4f\n\n", ...
        modelNames{i}, train_r2, test_r2, overfit_gap, verdict, test_rmse, test_mae);

    % Safe name for files
    safeName = strrep(modelNames{i}, " ", "_");

    %% Figure 1: Predicted vs Actual
    minVal = min([yTest; yPredTest]);
    maxVal = max([yTest; yPredTest]);

    f1 = figure('Visible','on');
    scatter(yTest, yPredTest, 18, 'filled');
    hold on;
    plot([minVal maxVal], [minVal maxVal], 'r--', 'LineWidth', 1.5);
    grid on;
    xlabel("Actual Final Percentage");
    ylabel("Predicted Final Percentage");
    title(modelNames{i} + " - Predicted vs Actual (Test R^2 = " + num2str(test_r2, "%.3f") + ")");
    hold off;
    saveas(f1, fullfile(figFolder, safeName + "_pred_vs_actual.png"));

    %% Figure 2: Residual Plot
    residuals = yTest - yPredTest;

    f2 = figure('Visible','on');
    scatter(yPredTest, residuals, 18, 'filled');
    hold on;
    yline(0, 'r--', 'LineWidth', 1.5);
    grid on;
    xlabel("Predicted Final Percentage");
    ylabel("Residuals");
    title(modelNames{i} + " - Residual Plot");
    hold off;
    saveas(f2, fullfile(figFolder, safeName + "_residuals.png"));

end

%% Sort results by Test RMSE
Results = sortrows(Results, "Test_RMSE");

disp(" ");
disp("===== SORTED RESULTS (best Test RMSE first) =====");
disp(Results);

%% Bar chart: Train R2 vs Test R2 for all models
f3 = figure('Visible','on');
modelLabels = Results.Model;
x = categorical(modelLabels, modelLabels);  % preserve sort order
bar(x, [Results.Train_R2, Results.Test_R2]);
legend("Train R²", "Test R²", "Location", "southwest");
ylabel("R²");
title("Train vs Test R² by Model");
grid on;
saveas(f3, fullfile(figFolder, "train_vs_test_R2_comparison.png"));


%% Save results table
writetable(Results, "regression_model_results.csv");
fprintf("\nResults saved to regression_model_results.csv\n");
fprintf("Figures saved in folder: %s\n", figFolder);
fprintf("Overfitting comparison chart saved: train_vs_test_R2_comparison.png\n");

%% Show best model
bestModelName = Results.Model(1);
fprintf("\nBest model (lowest Test RMSE): %s\n", bestModelName);

%% ===== Correlation Heatmap (NEW - does NOT affect your models) =====

% Create a copy so we don't touch your training pipeline
heatmapTable = T;

% Remove target if you want (optional)
% heatmapTable.(target) = [];

% Convert categorical variables to numeric for correlation
vars = heatmapTable.Properties.VariableNames;

for i = 1:length(vars)
    if iscategorical(heatmapTable.(vars{i}))
        heatmapTable.(vars{i}) = grp2idx(heatmapTable.(vars{i}));
    end
end

% Keep only numeric columns
isNum = varfun(@isnumeric, heatmapTable, 'OutputFormat', 'uniform');
numericData = heatmapTable(:, isNum);

% Compute correlation
corrMatrix = corr(table2array(numericData), 'Rows', 'complete');

% Plot heatmap
f4 = figure('Visible','on');
heatmap(numericData.Properties.VariableNames, ...
        numericData.Properties.VariableNames, ...
        corrMatrix);

title("Correlation Heatmap of Variables");

% Save it like your other figures
saveas(f4, fullfile(figFolder, "correlation_heatmap.png"));
