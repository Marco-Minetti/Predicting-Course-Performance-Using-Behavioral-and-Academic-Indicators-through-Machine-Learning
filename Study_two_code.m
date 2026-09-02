%% Board 38 - NSCG 2023 Graduate Outcomes ML Pipeline
% Dataset: National Survey of College Graduates 2023 (epcg23.csv)
%
% Research question:
% Can machine learning predict graduate employment status and salary
% based on education, demographics, and job characteristics?
%
% What this script DOES:
% 1) Predict employment status (employed vs not) - classification, 5 models
% 2) Feature importance for employment prediction
% 3) Ablation study - which feature groups matter most
% 4) Employment risk scores per graduate
% 5) Two-stage salary prediction for employed graduates - 6 regression models
% 6) Save all results, figures, and summary report
%
% What this script DOES NOT claim:
% - It does not make causal claims
% - Salary predictions are estimates, not guarantees
%
% Required: Statistics and Machine Learning Toolbox
% Put this file in the same folder as: epcg23.csv

clear; clc; close all;
rng(42);

%% =========================
% User settings
%% =========================
filename   = "epcg23.csv";
resultsFolder = "board38_nscg_results";

% Fast mode for development. Set false for final results.
fastMode = true;

if fastMode
    kFolds   = 3;
    numTrees = 60;
else
    kFolds   = 5;
    numTrees = 200;
end

fprintf("Fast mode: %d\n", fastMode);
fprintf("Cross-validation folds: %d\n", kFolds);
fprintf("Trees per ensemble: %d\n\n", numTrees);

if ~exist(resultsFolder, "dir")
    mkdir(resultsFolder);
end

%% =========================
% Load and prepare data
%% =========================
fprintf("Loading NSCG 2023 dataset...\n");
opts = detectImportOptions(filename);
opts.ExtraColumnsRule = "ignore";
Traw = readtable(filename, opts);

fprintf("Raw rows: %d | Columns: %d\n", height(Traw), width(Traw));

% Keep only the columns we need for modeling
% Employment target + features for classification
% Salary target + features for regression
keepCols = {'OBSNUM','AGE','SEX_2023','DGRDG','LFSTAT','WRKG', ...
            'OCEDRLP','EMSECSM','HRSWK','JOBSATIS', ...
            'SALARY','RACETHM','MARSTA','EDDAD','EDMOM','HDACYR', ...
            'NDGMEMG','WKSLYR'};

% Only keep columns that exist in the dataset
keepCols = keepCols(ismember(keepCols, Traw.Properties.VariableNames));
T = Traw(:, keepCols);

% Keep OBSNUM as ID, then remove from modeling
if ismember('OBSNUM', T.Properties.VariableNames)
    gradIDs = T.OBSNUM;
    T.OBSNUM = [];
else
    gradIDs = (1:height(T))';
end

% Remove logical skip values ('L') — these mean the question didn't apply
% (e.g. job-related questions for non-employed respondents)
catColsToClean = {'OCEDRLP','EMSECSM','JOBSATIS','WKSYR'};
for c = catColsToClean
    if ismember(c{1}, T.Properties.VariableNames)
        if iscell(T.(c{1})) || isstring(T.(c{1}))
            skipMask = strcmp(string(T.(c{1})), 'L');
            T.(c{1})(skipMask) = {'99'};
        end
    end
end

% Remove coded salary skips (9999998 = logical skip for non-employed)
if ismember('SALARY', T.Properties.VariableNames)
    T.SALARY(T.SALARY >= 9999990) = NaN;
end

% Convert string/cell columns to categorical
vars = string(T.Properties.VariableNames);
for v = vars
    col = T.(v);
    if iscell(col) || isstring(col)
        T.(v) = categorical(col);
    end
end

% NDGMEMG: field of study for highest degree (major group) — 7 nominal codes
% 1=Computer/math, 2=Bio/agri/env life sci, 3=Physical sci, 4=Social sci,
% 5=Engineering, 6=S&E-related, 7=Non-S&E (verify labels in Fpcg23.sas)
if ismember('NDGMEMG', T.Properties.VariableNames)
    T.NDGMEMG = categorical(T.NDGMEMG);
end

% Convert WRKG to binary numeric: Y=1, N=0
if ismember('WRKG', T.Properties.VariableNames)
    T.WRKG_BIN = double(string(T.WRKG) == 'Y');
    T.WRKG = [];
end

% Convert LFSTAT to categorical
if ismember('LFSTAT', T.Properties.VariableNames)
    T.LFSTAT = categorical(T.LFSTAT);
end

fprintf("Rows after prep: %d\n\n", height(T));

%% =========================
% Build classification table (employment prediction)
% Target: WRKG_BIN (1=employed, 0=not employed)
% Only use features known BEFORE employment — no job-related columns
%% =========================
empTarget = "WRKG_BIN";

% Remove columns that are only known AFTER/DURING employment (leakage)
% SALARY   — only known if employed
% LFSTAT   — encodes employment status directly
% HRSWK    — hours worked per week, only known if employed
% OCEDRLP  — relationship of job to degree, only known if employed
% EMSECSM  — employment sector, only known if employed
% JOBSATIS — job satisfaction, only known if employed
leakageCols = {'SALARY','LFSTAT','HRSWK','OCEDRLP','EMSECSM','JOBSATIS','WKSLYR'};
Tcls = T;
for lc = leakageCols
    if ismember(lc{1}, Tcls.Properties.VariableNames)
        Tcls.(lc{1}) = [];
    end
end

% Features remaining for classification:
% AGE, SEX_2023, DGRDG, RACETHM, MARSTA, EDDAD, EDMOM
% These are all known before employment status

% Remove rows missing the target
Tcls = Tcls(~isnan(Tcls.(empTarget)), :);
Tcls.(empTarget) = categorical(Tcls.(empTarget));

% Save class distribution
classDist = groupcounts(Tcls, empTarget);
writetable(classDist, fullfile(resultsFolder, "employment_class_distribution.csv"));
fprintf("Employment class distribution:\n");
disp(classDist);

%% ============================================================
% TASK 1: Employment status prediction (classification)
%% ============================================================
fprintf("\n===== TASK 1: EMPLOYMENT STATUS PREDICTION =====\n");

[empResults, empPred, empTruth] = classification_cv( ...
    Tcls, empTarget, kFolds, numTrees, resultsFolder, "employment");

writetable(empResults, fullfile(resultsFolder, "employment_cv_results.csv"));

%% ============================================================
% TASK 1b: Employment prediction, working-age adults only (22-60)
% Investigates whether AGE's dominance in Task 1 reflects genuine
% employability signal or is substantially driven by retirement
% among older respondents in the full sample.
%% ============================================================
fprintf("\n===== TASK 1b: EMPLOYMENT PREDICTION (WORKING AGE 22-60 ONLY) =====\n");

TclsWorking = Tcls(Tcls.AGE >= 22 & Tcls.AGE <= 60, :);

workingClassDist = groupcounts(TclsWorking, empTarget);
fprintf("Working-age (22-60) rows: %d\n", height(TclsWorking));
fprintf("Working-age class distribution:\n");
disp(workingClassDist);
writetable(workingClassDist, fullfile(resultsFolder, "employment_workingage_class_distribution.csv"));

[empResultsWorking, empPredWorking, empTruthWorking] = classification_cv( ...
    TclsWorking, empTarget, kFolds, numTrees, resultsFolder, "employment_workingage");

writetable(empResultsWorking, fullfile(resultsFolder, "employment_workingage_cv_results.csv"));

% Feature importance restricted to working-age sample, to see whether
% AGE's rank drops relative to the full-sample result once retirees
% are excluded.
fprintf("\n----- Feature importance (working-age only) -----\n");
[importanceTableWorking, ~] = compute_feature_importance( ...
    TclsWorking, empTarget, numTrees, resultsFolder, "employment_workingage");
disp(importanceTableWorking);
writetable(importanceTableWorking, fullfile(resultsFolder, "employment_workingage_feature_importance.csv"));

% Side-by-side comparison: full sample vs working-age-only
fprintf("\n----- Full sample vs Working-age comparison -----\n");
fprintf("Full sample    : Best model %s | Test Acc %.4f | Test F1 %.4f\n", ...
    empResults.Model(1), empResults.Test_Acc_Mean(1), empResults.Test_F1_Mean(1));
fprintf("Working age    : Best model %s | Test Acc %.4f | Test F1 %.4f\n", ...
    empResultsWorking.Model(1), empResultsWorking.Test_Acc_Mean(1), empResultsWorking.Test_F1_Mean(1));

% Single-feature comparison within working-age sample: does AGE alone still
% match HDACYR alone once retirees are excluded, mirroring the full-sample
% "Age Only (TRUE)" ablation result?
fprintf("\n----- AGE-only vs YearsSinceDegree-only, working-age sample -----\n");
ablationGroupsWorking = {
    "All Features",                    strings(1,0);
    "Age Only (TRUE)",                 ["SEX_2023","DGRDG","RACETHM","MARSTA","EDDAD","EDMOM","HDACYR","NDGMEMG"];
    "YearsSinceDegree Only (TRUE)",     ["AGE","SEX_2023","DGRDG","RACETHM","MARSTA","EDDAD","EDMOM","NDGMEMG"];
    "Remove Age AND YearsSinceDegree",  ["AGE","HDACYR"]
};
ablationResultsWorking = run_ablation_study( ...
    TclsWorking, empTarget, kFolds, numTrees, resultsFolder, ablationGroupsWorking);
disp(ablationResultsWorking);
writetable(ablationResultsWorking, fullfile(resultsFolder, "employment_workingage_ablation_results.csv"));

%% ============================================================
% TASK 2: Feature importance for employment
%% ============================================================
fprintf("\n===== TASK 2: FEATURE IMPORTANCE (EMPLOYMENT) =====\n");

[importanceTable, ~] = compute_feature_importance( ...
    Tcls, empTarget, numTrees, resultsFolder, "employment");

writetable(importanceTable, fullfile(resultsFolder, "employment_feature_importance.csv"));
disp(importanceTable);

%% ============================================================
% TASK 3: Ablation study
%% ============================================================
fprintf("\n===== TASK 3: ABLATION STUDY =====\n");

% Define feature groups to test
% Note: job-related columns already removed to prevent leakage
% Only pre-employment features remain
% Full feature set: AGE, SEX_2023, DGRDG, RACETHM, MARSTA, EDDAD, EDMOM, HDACYR, NDGMEMG
%
% AGE and HDACYR (years since highest degree) are highly correlated (r ~ 0.87)
% and both act partly as retirement proxies for older respondents. The groups
% below test each in isolation, each removed individually, and both removed
% together, to separate genuine independent signal from redundant/correlated
% "seniority" signal shared between the two.
ablationGroups = {
    "All Features",              strings(1,0);
    "Remove Demographics",       ["AGE","SEX_2023","RACETHM","MARSTA"];
    "Remove Education",          ["DGRDG","EDDAD","EDMOM"];
    "Remove Parental Edu",       ["EDDAD","EDMOM"];
    "Age Only (TRUE)",           ["SEX_2023","DGRDG","RACETHM","MARSTA","EDDAD","EDMOM","HDACYR","NDGMEMG"];
    "Remove Age Only",           ["AGE"];
    "YearsSinceDegree Only (TRUE)", ["AGE","SEX_2023","DGRDG","RACETHM","MARSTA","EDDAD","EDMOM","NDGMEMG"];
    "Remove YearsSinceDegree Only", ["HDACYR"];
    "Remove Age AND YearsSinceDegree", ["AGE","HDACYR"]
};

ablationResults = run_ablation_study( ...
    Tcls, empTarget, kFolds, numTrees, resultsFolder, ablationGroups);

writetable(ablationResults, fullfile(resultsFolder, "employment_ablation_results.csv"));
disp(ablationResults);

%% ============================================================
% TASK 4: Employment risk scores
%% ============================================================
fprintf("\n===== TASK 4: EMPLOYMENT RISK SCORES =====\n");

riskTable = create_risk_scores( ...
    Tcls, empTarget, gradIDs(1:height(Tcls)), kFolds, numTrees);

writetable(riskTable, fullfile(resultsFolder, "employment_risk_scores.csv"));
disp(riskTable(1:min(10, height(riskTable)), :));

%% ============================================================
% TASK 5: Two-stage salary prediction (employed graduates only)
%% ============================================================
fprintf("\n===== TASK 5: SALARY PREDICTION (EMPLOYED GRADUATES ONLY) =====\n");

% Build salary dataset: keep only employed with valid salary
% For salary prediction it is safe to include job-related features
% because we already know the person is employed at this stage
empMask = string(T.(empTarget)) == '1';
Tsal = T(empMask, :);
Tsal.(empTarget) = [];
if ismember('LFSTAT', Tsal.Properties.VariableNames)
    Tsal.LFSTAT = [];
end

% Convert job-related categoricals to numeric for regression models
% EMSECSM: 1=educational institution, 2=government, 3=business/industry
% OCEDRLP: 1=closely related, 2=somewhat related, 3=not related
% JOBSATIS: 1=very satisfied, 2=satisfied, 3=dissatisfied, 4=very dissatisfied
numericJobCols = {'OCEDRLP','EMSECSM','JOBSATIS'};
for c = numericJobCols
    if ismember(c{1}, Tsal.Properties.VariableNames)
        col = double(string(Tsal.(c{1})));
        col(isnan(col)) = NaN; % keep missing as NaN
        Tsal.(c{1}) = col;
    end
end

% Remove rows with missing or invalid salary
Tsal = Tsal(~isnan(Tsal.SALARY), :);
Tsal = Tsal(Tsal.SALARY > 0, :);
Tsal = rmmissing(Tsal);

% Log-transform salary to reduce skew and improve model performance
% Models will predict log(salary); we report back-transformed RMSE
Tsal.LOG_SALARY = log(Tsal.SALARY);
Tsal.SALARY = [];

fprintf("Employed graduates with valid salary: %d\n", height(Tsal));
fprintf("Salary features (%d total): %s\n", ...
    width(Tsal)-1, strjoin(setdiff(string(Tsal.Properties.VariableNames), "LOG_SALARY"), ", "));

salTarget = "LOG_SALARY";
ysal = Tsal.(salTarget);
Xsal = removevars(Tsal, salTarget);

% 50/50 shuffled split
nsal = height(Tsal);
salIdx = randperm(nsal);
salSplit = floor(nsal/2);

XsalTrain = Xsal(salIdx(1:salSplit), :);
XsalTest  = Xsal(salIdx(salSplit+1:end), :);
ysalTrain = ysal(salIdx(1:salSplit));
ysalTest  = ysal(salIdx(salSplit+1:end));

fprintf("Salary train rows: %d | test rows: %d\n\n", height(XsalTrain), numel(ysalTest));

% Train regression models for salary prediction
salModels     = {};
salModelNames = {};

% Linear Regression using fitlm — handles mixed categorical/numeric features properly
try
    trainDataLM = XsalTrain;
    trainDataLM.Response = ysalTrain;
    lmMdl = fitlm(trainDataLM);
    salModels{end+1} = lmMdl;
    salModelNames{end+1} = "Linear Regression";
catch ME
    fprintf("Skipping Linear Regression: %s\n", ME.message);
end
try
    salModels{end+1} = fitrtree(XsalTrain, ysalTrain);
    salModelNames{end+1} = "Regression Tree";
catch ME
    fprintf("Skipping Regression Tree: %s\n", ME.message);
end
try
    salModels{end+1} = fitrensemble(XsalTrain, ysalTrain, ...
        "Method","Bag","NumLearningCycles",numTrees);
    salModelNames{end+1} = "Bagged Trees";
catch ME
    fprintf("Skipping Bagged Trees: %s\n", ME.message);
end
try
    salModels{end+1} = fitrensemble(XsalTrain, ysalTrain, ...
        "Method","LSBoost","NumLearningCycles",numTrees);
    salModelNames{end+1} = "LSBoost";
catch ME
    fprintf("Skipping LSBoost: %s\n", ME.message);
end
try
    salModels{end+1} = fitrsvm(XsalTrain, ysalTrain, ...
        "KernelFunction","gaussian","Standardize",true);
    salModelNames{end+1} = "SVR";
catch ME
    fprintf("Skipping SVR: %s\n", ME.message);
end

% Note: GPR skipped — computationally fails on datasets this large (39k+ rows)

numSalModels = numel(salModels);

SalResults = table('Size',[numSalModels 8], ...
    'VariableTypes',["string","double","double","double","double","double","double","double"], ...
    'VariableNames',["Model","Train_RMSE","Train_MAE","Train_R2", ...
                     "Test_RMSE","Test_MAE","Test_R2","Overfit_Gap"]);

fprintf("===== SALARY MODEL RESULTS (Train vs Test, log-transformed target) =====\n");

for i = 1:numSalModels
    % fitlm predict takes a table; other models take a table too — both compatible
    yPredTest  = predict(salModels{i}, XsalTest);
    yPredTrain = predict(salModels{i}, XsalTrain);

    % Ensure column vectors
    yPredTest  = yPredTest(:);
    yPredTrain = yPredTrain(:);

    % R2 on log scale
    test_r2    = 1 - sum((yPredTest  - ysalTest).^2)  / sum((ysalTest  - mean(ysalTest)).^2);
    train_r2   = 1 - sum((yPredTrain - ysalTrain).^2) / sum((ysalTrain - mean(ysalTrain)).^2);
    overfit_gap = train_r2 - test_r2;

    % Back-transform to dollars for interpretable RMSE and MAE
    yPredTestDollar  = exp(yPredTest);
    yPredTrainDollar = exp(yPredTrain);
    ysalTestDollar   = exp(ysalTest);
    ysalTrainDollar  = exp(ysalTrain);

    test_rmse  = sqrt(mean((yPredTestDollar  - ysalTestDollar).^2));
    test_mae   = mean(abs(yPredTestDollar  - ysalTestDollar));
    train_rmse = sqrt(mean((yPredTrainDollar - ysalTrainDollar).^2));
    train_mae  = mean(abs(yPredTrainDollar - ysalTrainDollar));

    if overfit_gap < 0.02
        sal_verdict = "No overfitting";
    elseif overfit_gap < 0.05
        sal_verdict = "Mild";
    elseif overfit_gap < 0.10
        sal_verdict = "Moderate";
    else
        sal_verdict = "Likely overfitting";
    end

    SalResults.Model(i)       = string(salModelNames{i});
    SalResults.Train_RMSE(i)  = train_rmse;
    SalResults.Train_MAE(i)   = train_mae;
    SalResults.Train_R2(i)    = train_r2;
    SalResults.Test_RMSE(i)   = test_rmse;
    SalResults.Test_MAE(i)    = test_mae;
    SalResults.Test_R2(i)     = test_r2;
    SalResults.Overfit_Gap(i) = overfit_gap;

    fprintf("%s\n  Train R2: %.4f | Test R2: %.4f | Gap: %.4f => %s\n  Test RMSE: $%.0f | Test MAE: $%.0f\n\n", ...
        salModelNames{i}, train_r2, test_r2, overfit_gap, sal_verdict, test_rmse, test_mae);

    % Predicted vs Actual figure
    safeName = strrep(salModelNames{i}, " ", "_");
    minV = min([ysalTest; yPredTest]);
    maxV = max([ysalTest; yPredTest]);
    fs = figure("Visible","on");
    scatter(ysalTest, yPredTest, 8, "filled", "MarkerFaceAlpha", 0.3);
    hold on;
    plot([minV maxV],[minV maxV],"r--","LineWidth",1.5);
    grid on;
    xlabel("Actual Salary (USD)");
    ylabel("Predicted Salary (USD)");
    title("Salary - " + salModelNames{i} + " (Test R2 = " + num2str(test_r2,"%.3f") + ")");
    hold off;
    saveas(fs, fullfile(resultsFolder, "salary_" + safeName + "_pred_vs_actual.png"));
    close(fs);
end

SalResults = sortrows(SalResults, "Test_RMSE");
disp("===== SALARY RESULTS (best Test RMSE first) =====");
disp(SalResults);
writetable(SalResults, fullfile(resultsFolder, "salary_model_results.csv"));

% Train vs Test R2 bar chart for salary
f_sal = figure("Visible","on");
salLabels = SalResults.Model;
xsal = categorical(salLabels, salLabels);
bar(xsal, [SalResults.Train_R2, SalResults.Test_R2]);
legend("Train R2","Test R2","Location","southwest");
ylabel("R2");
title("Salary Prediction: Train vs Test R2 by Model");
grid on;
saveas(f_sal, fullfile(resultsFolder, "salary_train_vs_test_R2.png"));
close(f_sal);
fprintf("Salary results saved.\n");

%% ============================================================
% TASK 6: Career Efficiency Metric and Prediction
%% ============================================================
fprintf("\n===== TASK 6: CAREER EFFICIENCY METRIC =====\n");

% Concept: salary earned per unit of time and education invested
% Efficiency_raw = (hourly_rate / degree_investment) / sqrt(1 + years_since_degree)
%   hourly_rate       = SALARY / (HRSWK * 50 weeks)
%   degree_investment = 1 + 0.15*(DGRDG - 1)   (1=Bachelor 1.00, 2=Masters 1.15, 3=Doctorate 1.30, 4=Professional 1.45)
%   experience normalizer = sqrt(1 + years since highest degree)
% Final score is log-transformed and split into Low/Medium/High tertiles.

% Build efficiency dataset from employed graduates with valid salary
Teff = T(string(T.(empTarget)) == '1', :);
Teff = Teff(~isnan(Teff.SALARY) & Teff.SALARY > 0, :);
Teff = Teff(Teff.HRSWK > 0, :);
Teff = Teff(~isnan(Teff.HDACYR) & Teff.HDACYR > 1900, :);

% Compute efficiency components
% NSCG SALARY is an ANNUAL RATE, so divide by annualized hours (HRSWK x 52).
% WKSLYR (weeks worked last year) is intentionally NOT used here: dividing an
% annual rate by partial-year hours inflates hourly rates for part-year workers.
hourlyRate   = Teff.SALARY ./ (Teff.HRSWK * 52);
degreeInvest = 1 + 0.15 * (double(Teff.DGRDG) - 1);
yrsSinceDeg  = max(2023 - Teff.HDACYR, 0);
expNorm      = sqrt(1 + yrsSinceDeg);

effRaw = (hourlyRate ./ degreeInvest) ./ expNorm;
Teff.EFFICIENCY = log(effRaw);

% Discretize into tertiles: Low / Medium / High
q1 = quantile(Teff.EFFICIENCY, 1/3);
q2 = quantile(Teff.EFFICIENCY, 2/3);
effLevel = strings(height(Teff), 1);
effLevel(Teff.EFFICIENCY <  q1) = "Low";
effLevel(Teff.EFFICIENCY >= q1 & Teff.EFFICIENCY < q2) = "Medium";
effLevel(Teff.EFFICIENCY >= q2) = "High";
Teff.EFFICIENCY_LEVEL = categorical(effLevel);

fprintf("Graduates with efficiency score: %d\n", height(Teff));
fprintf("Efficiency score range: %.2f to %.2f (log scale)\n", ...
    min(Teff.EFFICIENCY), max(Teff.EFFICIENCY));
fprintf("Tertile cutoffs: Low < %.3f <= Medium < %.3f <= High\n\n", q1, q2);

% Efficiency distribution by degree level (research insight)
fprintf("Mean efficiency by degree level (1=Bach 2=Mast 3=Doctorate 4=Professional):\n");
effByDeg = groupsummary(Teff, "DGRDG", "mean", "EFFICIENCY");
disp(effByDeg);
writetable(effByDeg, fullfile(resultsFolder, "efficiency_by_degree.csv"));

% Efficiency by MAJOR FIELD (NDGMEMG) — which fields give the best career return?
% 1=Comp/math 2=Bio/life sci 3=Physical sci 4=Social sci 5=Engineering 6=S&E-related 7=Non-S&E
fprintf("Mean efficiency by major field of highest degree (NDGMEMG):\n");
effByMajor = groupsummary(Teff, "NDGMEMG", "mean", "EFFICIENCY");
effByMajor = sortrows(effByMajor, "mean_EFFICIENCY", "descend");
disp(effByMajor);
writetable(effByMajor, fullfile(resultsFolder, "efficiency_by_major.csv"));

% Bar chart of efficiency by major
f_maj = figure("Visible","on");
bar(effByMajor.mean_EFFICIENCY);
xticks(1:height(effByMajor));
xticklabels(string(effByMajor.NDGMEMG));
xlabel("Major field group (NDGMEMG code)");
ylabel("Mean Career Efficiency");
title("Career Efficiency by Field of Study of Highest Degree");
grid on;
saveas(f_maj, fullfile(resultsFolder, "efficiency_by_major.png"));
close(f_maj);

% Combined: efficiency by degree level WITHIN each major field
effByMajorDeg = groupsummary(Teff, ["NDGMEMG","DGRDG"], "mean", "EFFICIENCY");
writetable(effByMajorDeg, fullfile(resultsFolder, "efficiency_by_major_and_degree.csv"));

% Most efficient degree level PER major field
% Minimum cell size of 30 to avoid conclusions from tiny groups
% (e.g. Engineering x Professional has n=1 and must be excluded)
minCell = 30;
fprintf("\nMost efficient degree level per major field (cells with n >= %d):\n", minCell);
majorList = unique(effByMajorDeg.NDGMEMG);
bestRows = [];
for mi = 1:numel(majorList)
    sub = effByMajorDeg(effByMajorDeg.NDGMEMG == majorList(mi) & ...
                        effByMajorDeg.GroupCount >= minCell, :);
    if isempty(sub), continue; end
    [~, bi] = max(sub.mean_EFFICIENCY);
    bestRows = [bestRows; sub(bi,:)]; %#ok<AGROW>
end
BestDegPerMajor = bestRows;
BestDegPerMajor.Properties.VariableNames{'DGRDG'} = 'Best_Degree_Level';
disp(BestDegPerMajor);
writetable(BestDegPerMajor, fullfile(resultsFolder, "best_degree_per_major.csv"));

% Heatmap-style grouped bar chart: efficiency by degree within each major
f_md = figure("Visible","on");
majors = unique(effByMajorDeg.NDGMEMG);
degLevels = [1 2 3 4];
barData = nan(numel(majors), numel(degLevels));
for mi = 1:numel(majors)
    for di = 1:numel(degLevels)
        row = effByMajorDeg.NDGMEMG == majors(mi) & ...
              double(effByMajorDeg.DGRDG) == degLevels(di) & ...
              effByMajorDeg.GroupCount >= minCell;
        if any(row)
            barData(mi,di) = effByMajorDeg.mean_EFFICIENCY(row);
        end
    end
end
bar(barData);
xticks(1:numel(majors));
xticklabels(string(majors));
xlabel("Major field group (NDGMEMG code)");
ylabel("Mean Career Efficiency");
legend("Bachelor","Masters","Doctorate","Professional","Location","southoutside","Orientation","horizontal");
title("Career Efficiency by Degree Level within each Major Field");
grid on;
saveas(f_md, fullfile(resultsFolder, "efficiency_degree_within_major.png"));
close(f_md);

% --- Prediction features: pre-career background + degree field ---
% Exclude all formula components (SALARY, HRSWK, DGRDG, HDACYR) to avoid leakage
% NDGMEMG (field of study, major group) is chosen pre-career and NOT in the formula
effFeatures = {'AGE','SEX_2023','RACETHM','MARSTA','EDDAD','EDMOM','NDGMEMG'};
effFeatures = effFeatures(ismember(effFeatures, Teff.Properties.VariableNames));

% ============ Experiment A: Classify efficiency level ============
fprintf("\n--- Experiment A: Predict Efficiency Level (Low/Med/High) ---\n");
TeffCls = Teff(:, [effFeatures, {'EFFICIENCY_LEVEL'}]);
TeffCls = rmmissing(TeffCls);

[effClsResults, ~, ~] = classification_cv( ...
    TeffCls, "EFFICIENCY_LEVEL", kFolds, numTrees, resultsFolder, "efficiency_level");
writetable(effClsResults, fullfile(resultsFolder, "efficiency_level_cv_results.csv"));

% ============ Experiment B: Regress efficiency score ============
fprintf("\n--- Experiment B: Predict Efficiency Score (regression) ---\n");
TeffReg = Teff(:, [effFeatures, {'EFFICIENCY'}]);
TeffReg = rmmissing(TeffReg);

yEff = TeffReg.EFFICIENCY;
XEff = removevars(TeffReg, "EFFICIENCY");

nEff = height(TeffReg);
effIdx = randperm(nEff);
effSplit = floor(nEff/2);
XEffTrain = XEff(effIdx(1:effSplit), :);
XEffTest  = XEff(effIdx(effSplit+1:end), :);
yEffTrain = yEff(effIdx(1:effSplit));
yEffTest  = yEff(effIdx(effSplit+1:end));

effModels = {};
effModelNames = {};
try
    edata = XEffTrain; edata.Response = yEffTrain;
    effModels{end+1} = fitlm(edata);
    effModelNames{end+1} = "Linear Regression";
catch ME
    fprintf("Skipping Linear Regression: %s\n", ME.message);
end
try
    effModels{end+1} = fitrensemble(XEffTrain, yEffTrain, "Method","Bag","NumLearningCycles",numTrees);
    effModelNames{end+1} = "Bagged Trees";
catch ME
    fprintf("Skipping Bagged Trees: %s\n", ME.message);
end
try
    effModels{end+1} = fitrensemble(XEffTrain, yEffTrain, "Method","LSBoost","NumLearningCycles",numTrees);
    effModelNames{end+1} = "LSBoost";
catch ME
    fprintf("Skipping LSBoost: %s\n", ME.message);
end

numEffModels = numel(effModels);
EffResults = table('Size',[numEffModels 5], ...
    'VariableTypes',["string","double","double","double","double"], ...
    'VariableNames',["Model","Train_R2","Test_R2","Test_RMSE","Overfit_Gap"]);

for i = 1:numEffModels
    yp  = predict(effModels{i}, XEffTest);  yp  = yp(:);
    ypt = predict(effModels{i}, XEffTrain); ypt = ypt(:);

    tst_r2 = 1 - sum((yp  - yEffTest ).^2) / sum((yEffTest  - mean(yEffTest )).^2);
    trn_r2 = 1 - sum((ypt - yEffTrain).^2) / sum((yEffTrain - mean(yEffTrain)).^2);

    EffResults.Model(i)       = string(effModelNames{i});
    EffResults.Train_R2(i)    = trn_r2;
    EffResults.Test_R2(i)     = tst_r2;
    EffResults.Test_RMSE(i)   = sqrt(mean((yp - yEffTest).^2));
    EffResults.Overfit_Gap(i) = trn_r2 - tst_r2;

    fprintf("%s | Train R2: %.4f | Test R2: %.4f | Gap: %.4f\n", ...
        effModelNames{i}, trn_r2, tst_r2, trn_r2 - tst_r2);
end

EffResults = sortrows(EffResults, "Test_R2", "descend");
disp(EffResults);
writetable(EffResults, fullfile(resultsFolder, "efficiency_regression_results.csv"));

% Efficiency distribution histogram
f_eff = figure("Visible","on");
histogram(Teff.EFFICIENCY, 60);
xline(q1, 'r--', 'LineWidth', 1.5);
xline(q2, 'r--', 'LineWidth', 1.5);
xlabel("Career Efficiency (log scale)");
ylabel("Number of graduates");
title("Career Efficiency Distribution with Low/Medium/High cutoffs");
grid on;
saveas(f_eff, fullfile(resultsFolder, "efficiency_distribution.png"));
close(f_eff);

fprintf("Efficiency results saved.\n");

%% ============================================================
% Final summary report
%% ============================================================
write_summary_report(resultsFolder, empResults, importanceTable, ...
    ablationResults, classDist, SalResults, fastMode, kFolds, numTrees);

fprintf("\nDONE. All results saved in: %s\n", resultsFolder);

%% ========================================================================
% LOCAL FUNCTIONS
%% ========================================================================

function [Results, bestPred, yAll] = classification_cv(T, target, k, numTrees, resultsFolder, prefix)
    T = rmmissing(T);
    yAll = T.(target);
    if ~iscategorical(yAll)
        yAll = categorical(yAll);
        T.(target) = yAll;
    end

    cv = cvpartition(height(T), "KFold", k);
    modelNames  = ["Decision Tree","Bagged Trees","LSBoost","SVM","Logistic Regression"];
    numModels   = numel(modelNames);

    accAll      = zeros(k, numModels);
    f1All       = zeros(k, numModels);
    trainAccAll = zeros(k, numModels);
    trainF1All  = zeros(k, numModels);
    predOOF     = strings(height(T), numModels);

    for fold = 1:k
        trainIdx = training(cv, fold);
        testIdx  = test(cv, fold);

        XTrain = removevars(T(trainIdx,:), target);
        yTrain = T.(target)(trainIdx);
        XTest  = removevars(T(testIdx,:),  target);
        yTest  = T.(target)(testIdx);

        models = cell(1, numModels);
        models{1} = fitctree(XTrain, yTrain);
        models{2} = fitcensemble(XTrain, yTrain, ...
            "Method","Bag","NumLearningCycles",numTrees);
        try
            if numel(categories(yTrain)) > 2
                models{3} = fitcensemble(XTrain, yTrain, ...
                    "Method","AdaBoostM2","NumLearningCycles",numTrees);
            else
                models{3} = fitcensemble(XTrain, yTrain, ...
                    "Method","AdaBoostM1","NumLearningCycles",numTrees);
            end
        catch
            models{3} = fitctree(XTrain, yTrain);
        end

        % SVM and Logistic Regression only support 2 classes natively;
        % use fitcecoc (error-correcting output codes) for 3+ classes
        isMulticlass = numel(categories(yTrain)) > 2;
        try
            if isMulticlass
                svmTemplate = templateSVM("KernelFunction","gaussian","Standardize",true);
                models{4} = fitcecoc(XTrain, yTrain, "Learners", svmTemplate);
            else
                models{4} = fitcsvm(XTrain, yTrain, ...
                    "KernelFunction","gaussian","Standardize",true);
            end
        catch
            models{4} = fitctree(XTrain, yTrain);
        end
        try
            if isMulticlass
                linTemplate = templateLinear("Learner","logistic");
                models{5} = fitcecoc(XTrain, yTrain, "Learners", linTemplate);
            else
                models{5} = fitclinear(XTrain, yTrain, "Learner","logistic");
            end
        catch
            models{5} = fitctree(XTrain, yTrain);
        end

        for m = 1:numModels
            yPred      = predict(models{m}, XTest);
            accAll(fold,m)  = mean(yPred == yTest);
            f1All(fold,m)   = macro_f1(yTest, yPred);
            predOOF(testIdx,m) = string(yPred);

            yPredTrain = predict(models{m}, XTrain);
            trainAccAll(fold,m) = mean(yPredTrain == yTrain);
            trainF1All(fold,m)  = macro_f1(yTrain, yPredTrain);
        end
        fprintf("Fold %d complete.\n", fold);
    end

    verdicts = strings(numModels,1);
    for i = 1:numModels
        g = mean(trainAccAll(:,i)) - mean(accAll(:,i));
        if g < 0.02,      verdicts(i) = "No overfitting";
        elseif g < 0.05,  verdicts(i) = "Mild";
        elseif g < 0.10,  verdicts(i) = "Moderate";
        else,             verdicts(i) = "Likely overfitting";
        end
    end

    Results = table;
    Results.Model          = modelNames';
    Results.Train_Acc_Mean = mean(trainAccAll,1)';
    Results.Test_Acc_Mean  = mean(accAll,1)';
    Results.Overfit_Gap    = mean(trainAccAll,1)' - mean(accAll,1)';
    Results.Train_F1_Mean  = mean(trainF1All,1)';
    Results.Test_F1_Mean   = mean(f1All,1)';
    Results.Accuracy_Std   = std(accAll,0,1)';
    Results.Verdict        = verdicts;
    Results = sortrows(Results, "Test_Acc_Mean", "descend");

    fprintf("\n===== OVERFITTING CHECK =====\n");
    for i = 1:height(Results)
        fprintf("%s\n  Train Acc: %.4f | Test Acc: %.4f | Gap: %.4f => %s\n  Train F1: %.4f | Test F1: %.4f\n\n", ...
            Results.Model(i), Results.Train_Acc_Mean(i), Results.Test_Acc_Mean(i), ...
            Results.Overfit_Gap(i), Results.Verdict(i), ...
            Results.Train_F1_Mean(i), Results.Test_F1_Mean(i));
    end
    disp(Results);

    % Confusion matrix for best model
    bestIdx = find(modelNames == Results.Model(1), 1);
    bestPred = categorical(predOOF(:,bestIdx), categories(yAll));
    fig = figure("Visible","off");
    confusionchart(yAll, bestPred);
    title(prefix + " - CV Confusion Matrix (" + Results.Model(1) + ")");
    saveas(fig, fullfile(resultsFolder, prefix + "_confusion_matrix.png"));
    close(fig);
end

function [importanceTable, mdl] = compute_feature_importance(T, target, numTrees, resultsFolder, prefix)
    T = rmmissing(T);
    y = T.(target);
    if ~iscategorical(y)
        y = categorical(y);
        T.(target) = y;
    end
    X = removevars(T, target);
    mdl = fitcensemble(X, y, "Method","Bag","NumLearningCycles",numTrees);
    imp = predictorImportance(mdl);
    predictors = string(mdl.PredictorNames);
    predictors = predictors(:);
    imp = imp(:);
    importanceTable = table(predictors, imp, 'VariableNames', {'Predictor','Importance'});
    importanceTable = sortrows(importanceTable, "Importance", "descend");

    fig = figure("Visible","off");
    bar(importanceTable.Importance);
    xticks(1:height(importanceTable));
    xticklabels(importanceTable.Predictor);
    xtickangle(45);
    ylabel("Importance");
    title(prefix + " Feature Importance");
    grid on;
    saveas(fig, fullfile(resultsFolder, prefix + "_feature_importance.png"));
    close(fig);
end

function Results = run_ablation_study(T, target, k, numTrees, resultsFolder, groups)
    nGroups = size(groups, 1);
    names = strings(nGroups, 1);
    acc   = zeros(nGroups, 1);
    f1    = zeros(nGroups, 1);

    for i = 1:nGroups
        names(i)      = groups{i,1};
        varsToRemove  = groups{i,2};
        Ttmp = T;
        for v = varsToRemove
            if ismember(v, string(Ttmp.Properties.VariableNames))
                Ttmp.(v) = [];
            end
        end
        [acc(i), f1(i)] = quick_cv(Ttmp, target, k, numTrees);
        fprintf("%s | Accuracy %.4f | MacroF1 %.4f\n", names(i), acc(i), f1(i));
    end

    Results = table(names(:), acc(:), f1(:), ...
        'VariableNames', {'Experiment','Accuracy','MacroF1'});
    Results = sortrows(Results, 'Accuracy', 'descend');

    fig = figure("Visible","off");
    bar(Results.Accuracy);
    xticks(1:height(Results));
    xticklabels(Results.Experiment);
    xtickangle(45);
    ylabel("CV Accuracy");
    title("Ablation Study: Employment Prediction");
    grid on;
    saveas(fig, fullfile(resultsFolder, "employment_ablation_accuracy.png"));
    close(fig);
end

function [accMean, f1Mean] = quick_cv(T, target, k, numTrees)
    T = rmmissing(T);
    y = T.(target);
    if ~iscategorical(y)
        y = categorical(y);
        T.(target) = y;
    end
    cv  = cvpartition(height(T), "KFold", k);
    acc = zeros(k,1);
    f1  = zeros(k,1);
    for fold = 1:k
        XTrain = removevars(T(training(cv,fold),:), target);
        yTrain = T.(target)(training(cv,fold));
        XTest  = removevars(T(test(cv,fold),:),     target);
        yTest  = T.(target)(test(cv,fold));
        mdl    = fitcensemble(XTrain, yTrain, "Method","Bag","NumLearningCycles",numTrees);
        yPred  = predict(mdl, XTest);
        acc(fold) = mean(yPred == yTest);
        f1(fold)  = macro_f1(yTest, yPred);
    end
    accMean = mean(acc);
    f1Mean  = mean(f1);
end

function riskTable = create_risk_scores(T, target, IDs, k, numTrees)
    T = rmmissing(T);
    y = T.(target);
    if ~iscategorical(y)
        y = categorical(y);
        T.(target) = y;
    end
    cv      = cvpartition(height(T), "KFold", k);
    pEmploy = zeros(height(T), 1);

    for fold = 1:k
        XTrain = removevars(T(training(cv,fold),:), target);
        yTrain = T.(target)(training(cv,fold));
        XTest  = removevars(T(test(cv,fold),:),     target);
        mdl    = fitcensemble(XTrain, yTrain, "Method","Bag","NumLearningCycles",numTrees);
        [~, score] = predict(mdl, XTest);
        classNames = string(mdl.ClassNames);
        idx1 = find(classNames == "1", 1);
        if isempty(idx1), idx1 = numel(classNames); end
        pEmploy(test(cv,fold)) = score(:, idx1);
    end

    riskScore = round(100 * (1 - pEmploy), 2);
    riskGroup = strings(height(T), 1);
    riskGroup(riskScore <  30) = "Low_Risk";
    riskGroup(riskScore >= 30 & riskScore < 70) = "Medium_Risk";
    riskGroup(riskScore >= 70) = "High_Risk";

    riskTable = table(IDs(1:height(T)), string(T.(target)), ...
        round(pEmploy,4), riskScore(:), categorical(riskGroup(:)), ...
        'VariableNames', {'Graduate_ID','Actual_Employment', ...
        'Predicted_Prob_Employed','Employment_Risk_Score','Risk_Group'});
end

function f1 = macro_f1(yTrue, yPred)
    if ~iscategorical(yTrue), yTrue = categorical(yTrue); end
    if ~iscategorical(yPred), yPred = categorical(yPred); end
    classes  = categories(yTrue);
    f1Scores = zeros(numel(classes), 1);
    for i = 1:numel(classes)
        c  = classes{i};
        tp = sum(yPred == c & yTrue == c);
        fp = sum(yPred == c & yTrue ~= c);
        fn = sum(yPred ~= c & yTrue == c);
        precision   = tp / max(tp+fp, 1);
        recall      = tp / max(tp+fn, 1);
        f1Scores(i) = 2 * precision * recall / max(precision+recall, 1e-12);
    end
    f1 = mean(f1Scores);
end

function write_summary_report(resultsFolder, empResults, importanceTable, ...
        ablationResults, classDist, SalResults, fastMode, kFolds, numTrees)

    fileID = fopen(fullfile(resultsFolder, "board38_nscg_summary_report.txt"), "w");
    fprintf(fileID, "Board 38 - NSCG 2023 Graduate Outcomes ML Summary\n");
    fprintf(fileID, "==================================================\n\n");

    fprintf(fileID, "Research question:\n");
    fprintf(fileID, "Can ML predict graduate employment status and salary from education and demographic features?\n\n");

    fprintf(fileID, "Settings: Fast mode=%d | Folds=%d | Trees=%d\n\n", fastMode, kFolds, numTrees);

    fprintf(fileID, "Class distribution (employment):\n");
    for i = 1:height(classDist)
        fprintf(fileID, "  %s: %d\n", string(classDist.WRKG_BIN(i)), classDist.GroupCount(i));
    end
    fprintf(fileID, "\n");

    fprintf(fileID, "1) Employment prediction\n");
    fprintf(fileID, "   Best model: %s\n", empResults.Model(1));
    fprintf(fileID, "   Test Accuracy: %.4f\n", empResults.Test_Acc_Mean(1));
    fprintf(fileID, "   Test MacroF1:  %.4f\n", empResults.Test_F1_Mean(1));
    fprintf(fileID, "   Overfit Gap:   %.4f => %s\n\n", empResults.Overfit_Gap(1), empResults.Verdict(1));

    fprintf(fileID, "2) Top employment predictors\n");
    for i = 1:min(5, height(importanceTable))
        fprintf(fileID, "   %d. %s (%.6f)\n", i, importanceTable.Predictor(i), importanceTable.Importance(i));
    end
    fprintf(fileID, "\n");

    fprintf(fileID, "3) Ablation study\n");
    for i = 1:height(ablationResults)
        fprintf(fileID, "   %s | Acc %.4f | F1 %.4f\n", ...
            ablationResults.Experiment(i), ablationResults.Accuracy(i), ablationResults.MacroF1(i));
    end
    fprintf(fileID, "\n");

    fprintf(fileID, "4) Salary prediction (employed graduates only)\n");
    fprintf(fileID, "   Best model: %s\n", SalResults.Model(1));
    fprintf(fileID, "   Test R2:    %.4f\n", SalResults.Test_R2(1));
    fprintf(fileID, "   Test RMSE:  $%.0f\n", SalResults.Test_RMSE(1));
    fprintf(fileID, "   Overfit Gap: %.4f\n\n", SalResults.Overfit_Gap(1));

    fprintf(fileID, "Notes:\n");
    fprintf(fileID, "- SALARY coded as 9999998 for non-employed; filtered before regression.\n");
    fprintf(fileID, "- Logical skip values (L) recoded as category 99 before modeling.\n");
    fprintf(fileID, "- Results are descriptive, not causal.\n");

    fclose(fileID);
end