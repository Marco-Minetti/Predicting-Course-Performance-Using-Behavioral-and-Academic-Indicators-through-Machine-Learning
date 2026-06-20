%% Board 38 - PISA 2022 Student Questionnaire ML Starter
% Goal:
% Use PISA 2022 student questionnaire data to predict student math performance.
%
% Put this file in the same folder as:
% pisa_2022_student_questionnaire.csv

clear; clc; close all;
rng(42);

%% =========================
% Settings
%% =========================
fileName = "pisa_2022_student_questionnaire.csv";
resultsFolder = "pisa_2022_results";

targetDomain = "MATH";     % options: "MATH", "READ", "SCIE"
kFolds = 3;                % use 3 for speed, 5 for final results
maxRows = 30000;           % reduce for speed if dataset is huge
maxPredictors = 60;        % keep model manageable

if ~exist(resultsFolder, "dir")
    mkdir(resultsFolder);
end

%% =========================
% Load data
%% =========================
T = readtable(fileName);

fprintf("Rows: %d\n", height(T));
fprintf("Columns: %d\n", width(T));

%% =========================
% Clean column names
%% =========================
T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);

%% =========================
% Save column names
%% =========================
columnTable = table(string(T.Properties.VariableNames)', ...
    'VariableNames', {'ColumnName'});

writetable(columnTable, fullfile(resultsFolder, "pisa_column_names.csv"));

disp("Column names saved to pisa_2022_results/pisa_column_names.csv");

%% =========================
% Optional: sample rows for speed
%% =========================
if height(T) > maxRows
    idx = randperm(height(T), maxRows);
    T = T(idx,:);
    fprintf("Sampled %d rows for faster training.\n", maxRows);
end

%% =========================
% Create target from plausible values
%% =========================
pvPrefix = "PV";
pvVars = strings(10,1);

for i = 1:10
    pvVars(i) = pvPrefix + i + targetDomain;
end

availablePV = pvVars(ismember(pvVars, string(T.Properties.VariableNames)));

if isempty(availablePV)
    error("No plausible value columns found for %s. Check column names in pisa_column_names.csv.", targetDomain);
end

fprintf("Using target columns:\n");
disp(availablePV);

% Average plausible values to create one simplified target
T.MathScore = mean(T{:, availablePV}, 2, "omitnan");

%% =========================
% Remove rows with missing target
%% =========================
T = rmmissing(T, "DataVariables", "MathScore");

%% =========================
% Build predictor table
%% =========================
X = T;

% Remove all plausible values to avoid leakage
allPV = startsWith(string(X.Properties.VariableNames), "PV");
X(:, allPV) = [];

% Remove target
X.MathScore = [];

% Remove obvious IDs / weights / technical columns
removePatterns = ["CNTSTUID", "CNTSCHID", "SCHOOLID", "STUDENTID", ...
                  "W_FSTUWT", "W_FSTURWT", "SENWT", "VER_", "BOOKID"];

vars = string(X.Properties.VariableNames);

toRemove = false(size(vars));

for p = removePatterns
    toRemove = toRemove | startsWith(vars, p);
end

X(:, toRemove) = [];

%% =========================
% Convert text variables to categorical
%% =========================
vars = string(X.Properties.VariableNames);

for v = vars
    col = X.(v);

    if iscellstr(col) || isstring(col)
        X.(v) = categorical(col);
    end
end

%% =========================
% Remove columns with too much missing data
%% =========================
missingRate = zeros(1,width(X));

for i = 1:width(X)
    missingRate(i) = mean(ismissing(X.(i)));
end

keepCols = missingRate < 0.40;
X = X(:, keepCols);

fprintf("Predictors after removing high-missing columns: %d\n", width(X));

%% =========================
% Limit number of predictors for speed
%% =========================
if width(X) > maxPredictors
    X = X(:, 1:maxPredictors);
    fprintf("Limited predictors to first %d columns for speed.\n", maxPredictors);
end

%% =========================
% Combine X and target
%% =========================
ML = X;
ML.MathScore = T.MathScore;

ML = rmmissing(ML);

fprintf("Rows used for ML after cleaning: %d\n", height(ML));
fprintf("Predictors used: %d\n", width(ML)-1);

%% =========================
% Cross-validation regression
%% =========================
target = "MathScore";

cv = cvpartition(height(ML), "KFold", kFolds);

modelNames = ["Linear Regression", "Regression Tree", "Bagged Trees", "LSBoost"];
numModels = numel(modelNames);

rmseAll = zeros(kFolds,numModels);
maeAll = zeros(kFolds,numModels);
r2All = zeros(kFolds,numModels);

predOOF = zeros(height(ML),numModels);
yAll = ML.(target);

fprintf("\n===== PISA %s SCORE PREDICTION =====\n", targetDomain);

for fold = 1:kFolds

    trainT = ML(training(cv,fold),:);
    testT  = ML(test(cv,fold),:);

    yTrain = trainT.(target);
    yTest  = testT.(target);

    XTrain = removevars(trainT,target);
    XTest  = removevars(testT,target);

    models = cell(1,numModels);

    models{1} = fitrlinear(XTrain,yTrain);

    models{2} = fitrtree(XTrain,yTrain);

    models{3} = fitrensemble(XTrain,yTrain, ...
        "Method","Bag", ...
        "NumLearningCycles",60);

    models{4} = fitrensemble(XTrain,yTrain, ...
        "Method","LSBoost", ...
        "NumLearningCycles",100);

    for m = 1:numModels

        yPred = predict(models{m},XTest);

        predOOF(test(cv,fold),m) = yPred;

        rmseAll(fold,m) = sqrt(mean((yPred-yTest).^2));
        maeAll(fold,m) = mean(abs(yPred-yTest));
        r2All(fold,m) = 1 - sum((yPred-yTest).^2) / sum((yTest-mean(yTest)).^2);

    end

    fprintf("Fold %d complete.\n", fold);
end

%% =========================
% Results table
%% =========================
Results = table;
Results.Model = modelNames';
Results.RMSE_Mean = mean(rmseAll,1)';
Results.RMSE_Std = std(rmseAll,0,1)';
Results.MAE_Mean = mean(maeAll,1)';
Results.MAE_Std = std(maeAll,0,1)';
Results.R2_Mean = mean(r2All,1)';
Results.R2_Std = std(r2All,0,1)';

Results = sortrows(Results, "RMSE_Mean");

disp(Results);

writetable(Results, fullfile(resultsFolder, "pisa_math_prediction_results.csv"));

%% =========================
% Best model figure
%% =========================
bestModelName = Results.Model(1);
bestIdx = find(modelNames == bestModelName, 1);

bestPred = predOOF(:,bestIdx);

fig1 = figure("Visible","on");
scatter(yAll,bestPred,10,"filled");
hold on;

minVal = min([yAll; bestPred]);
maxVal = max([yAll; bestPred]);

plot([minVal maxVal],[minVal maxVal],"r--","LineWidth",1.5);

grid on;
xlabel("Actual PISA Math Score");
ylabel("Predicted PISA Math Score");
title("PISA 2022 Math Prediction - " + bestModelName);
hold off;

saveas(fig1, fullfile(resultsFolder, "pisa_predicted_vs_actual.png"));

%% =========================
% Residual plot
%% =========================
residuals = yAll - bestPred;

fig2 = figure("Visible","on");
scatter(bestPred,residuals,10,"filled");
hold on;
yline(0,"r--","LineWidth",1.5);
grid on;
xlabel("Predicted Score");
ylabel("Residual");
title("PISA 2022 Residual Plot - " + bestModelName);
hold off;

saveas(fig2, fullfile(resultsFolder, "pisa_residuals.png"));

fprintf("\nDONE. Results saved in folder: %s\n", resultsFolder);