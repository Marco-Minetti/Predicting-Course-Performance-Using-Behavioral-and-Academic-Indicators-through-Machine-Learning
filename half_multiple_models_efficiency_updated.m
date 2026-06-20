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


%% Check that at least one model was trained
numModels = numel(models);
if numModels == 0
    error("No models were successfully trained.");
end

%% Results table — Train vs Test + Overfit Gap
Results = table('Size', [numModels 8], ...
    'VariableTypes', ["string","double","double","double","double","double","double","double"], ...
    'VariableNames', ["Model","Train_RMSE","Train_MAE","Train_R2","Test_RMSE","Test_MAE","Test_R2","Overfit_Gap"]);

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
    train_mae  = mean(abs(yPredTrain - yTrain));
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
    Results.Train_MAE(i)   = train_mae;
    Results.Train_R2(i)    = train_r2;
    Results.Test_RMSE(i)   = test_rmse;
    Results.Test_MAE(i)    = test_mae;
    Results.Test_R2(i)     = test_r2;
    Results.Overfit_Gap(i) = overfit_gap;

    fprintf("%s\n  Train R²: %.4f | Test R²: %.4f | Gap: %.4f => %s\n  Train RMSE: %.4f | Train MAE: %.4f\n  Test RMSE:  %.4f | Test MAE:  %.4f\n\n", ...
        modelNames{i}, train_r2, test_r2, overfit_gap, verdict, train_rmse, train_mae, test_rmse, test_mae);

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
    close(f1);

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
    close(f2);

end

%% Sort results by Test RMSE
Results = sortrows(Results, "Test_RMSE");

disp(" ");
disp("===== SORTED RESULTS (best Test RMSE first) =====");
disp(Results);

%% 5-Fold Cross Validation
fprintf("\n===== 5-FOLD CROSS VALIDATION =====\n");

k = 5;
Xall = removevars(T, target);
Yall = T.(target);
n_all = height(T);

% Create fold indices
cv = cvpartition(n_all, "KFold", k);

% CV model definitions: name + training function handle
cvModelDefs = {
    "Linear Regression",         @(X,y) fitrlinear(X, y);
    "Regression Tree",           @(X,y) fitrtree(X, y);
    "Bagged Trees",              @(X,y) fitrensemble(X, y, "Method", "Bag", "NumLearningCycles", 200);
    "LSBoost Ensemble",          @(X,y) fitrensemble(X, y, "Method", "LSBoost", "NumLearningCycles", 300);
    "Gaussian Process Regression", @(X,y) fitrgp(X, y);
    "Support Vector Regression", @(X,y) fitrsvm(X, y, "KernelFunction", "gaussian", "Standardize", true);
};

numCVModels = size(cvModelDefs, 1);

CV_Results = table('Size', [numCVModels 4], ...
    'VariableTypes', ["string","double","double","double"], ...
    'VariableNames', ["Model","CV_RMSE","CV_MAE","CV_R2"]);

for i = 1:numCVModels
    cvName    = cvModelDefs{i,1};
    cvTrainFn = cvModelDefs{i,2};

    fold_rmse = zeros(k,1);
    fold_mae  = zeros(k,1);
    fold_r2   = zeros(k,1);

    success = true;
    for f = 1:k
        try
            XcvTrain = Xall(training(cv,f), :);
            YcvTrain = Yall(training(cv,f));
            XcvTest  = Xall(test(cv,f), :);
            YcvTest  = Yall(test(cv,f));

            mdl = cvTrainFn(XcvTrain, YcvTrain);
            yhat = predict(mdl, XcvTest);

            fold_rmse(f) = sqrt(mean((yhat - YcvTest).^2));
            fold_mae(f)  = mean(abs(yhat - YcvTest));
            fold_r2(f)   = 1 - sum((yhat - YcvTest).^2) / sum((YcvTest - mean(YcvTest)).^2);
        catch ME
            fprintf("  Fold %d failed for %s: %s\n", f, cvName, ME.message);
            success = false;
            break;
        end
    end

    if success
        CV_Results.Model(i)   = cvName;
        CV_Results.CV_RMSE(i) = mean(fold_rmse);
        CV_Results.CV_MAE(i)  = mean(fold_mae);
        CV_Results.CV_R2(i)   = mean(fold_r2);

        fprintf("%s\n  CV RMSE: %.4f | CV MAE: %.4f | CV R²: %.4f\n\n", ...
            cvName, mean(fold_rmse), mean(fold_mae), mean(fold_r2));
    else
        CV_Results.Model(i) = cvName;
        fprintf("  Skipping %s due to fold error.\n\n", cvName);
    end
end

CV_Results = sortrows(CV_Results, "CV_RMSE");

disp("===== CV RESULTS (best first) =====");
disp(CV_Results);

writetable(CV_Results, "cv_model_results.csv");
fprintf("CV results saved to cv_model_results.csv\n");

%% Bar chart: Test R2 vs CV R2 comparison
% Align both result tables by model name
alignedTest_R2 = zeros(numCVModels, 1);
alignedCV_R2   = zeros(numCVModels, 1);
alignedNames   = CV_Results.Model;

for i = 1:numCVModels
    idx = find(Results.Model == CV_Results.Model(i));
    if ~isempty(idx)
        alignedTest_R2(i) = Results.Test_R2(idx);
    end
    alignedCV_R2(i) = CV_Results.CV_R2(i);
end

f4 = figure('Visible','on');
x4 = categorical(alignedNames, alignedNames);
bar(x4, [alignedTest_R2, alignedCV_R2]);
legend("Hold-out Test R²", "5-Fold CV R²", "Location", "southwest");
ylabel("R²");
title("Hold-out Test R² vs 5-Fold CV R² by Model");
grid on;
saveas(f4, fullfile(figFolder, "test_vs_cv_R2_comparison.png"));
close(f4);
fprintf("Test vs CV R² chart saved: test_vs_cv_R2_comparison.png\n");

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
close(f3);

%% Save results table
writetable(Results, "regression_model_results.csv");
fprintf("\nResults saved to regression_model_results.csv\n");
fprintf("Figures saved in folder: %s\n", figFolder);
fprintf("Overfitting comparison chart saved: train_vs_test_R2_comparison.png\n");

%% Show best model
bestModelName = Results.Model(1);
fprintf("\nBest model (lowest Test RMSE): %s\n", bestModelName);