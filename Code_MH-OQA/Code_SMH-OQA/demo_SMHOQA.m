clear all;
warning off; 
clc;

%% Dataset Loading 
load('D:\Hashing produce\datasets\mir_cnn.mat')
fprintf('MIR Flickr dataset loaded...\n');

%% Parameters Setting
run = 5;
bits = 128;
n_anchors = 1000;
map = zeros(run,1);
alpha = 1e-3;
beta = 1e-5;
delta = 1e-3;
lambda = 1e5;
maxIter = 10;

%% Preparing Dataset
fprintf('Preparing dataset...\n');
Ntrain = size(I_tr,1); 
sample = randsample(Ntrain, n_anchors);
anchorI = I_tr(sample,:);
anchorT = T_tr(sample,:);
sigmaI = 60;
sigmaT = 80;
PhiI = exp(-sqdist(I_tr,anchorI)/(2*sigmaI*sigmaI));
PhiI = [PhiI, ones(Ntrain,1)];
PhiT = exp(-sqdist(T_tr,anchorT)/(2*sigmaT*sigmaT));
PhiT = [PhiT, ones(Ntrain,1)];
Phi_testI = exp(-sqdist(I_te,anchorI)/(2*sigmaI*sigmaI));
Phi_testI = [Phi_testI, ones(size(Phi_testI,1),1)];
Phi_testT = exp(-sqdist(T_te,anchorT)/(2*sigmaT*sigmaT));
Phi_testT = [Phi_testT, ones(size(Phi_testT,1),1)];
Phi_dbI = exp(-sqdist(I_db,anchorI)/(2*sigmaI*sigmaI));
Phi_dbI = [Phi_dbI, ones(size(Phi_dbI,1),1)];
Phi_dbT = exp(-sqdist(T_db,anchorT)/(2*sigmaT*sigmaT));
Phi_dbT = [Phi_dbT, ones(size(Phi_dbT,1),1)];
PhiI = PhiI';
PhiT = PhiT';
Phi_testI = Phi_testI';
Phi_testT = Phi_testT';
Phi_dbI = Phi_dbI';
Phi_dbT = Phi_dbT';

%% Training & Evaluation Process
fprintf('\n============================================Start training SMH-OQA============================================\n');
% Seting parameters    
param.bits = bits;
param.maxIter = maxIter;
param.alpha = alpha;
param.beta = beta;
param.delta = delta;
param.lambda = lambda;
param.L = L_tr';

for j = 1 : run
    % Training model    
    [B_db, B_test] = solve_SMHOQA(PhiI, PhiT, Phi_testI, Phi_testT, Phi_dbI, Phi_dbT, param, j);
    % Evaluation  
    B_db = compactbit(B_db>0);
    B_test = compactbit(B_test>0);
    Dhamm = hammingDist(B_db+2, B_test+2);
    [MAP] = perf_metric4Label( L_db, L_te, Dhamm);
    map(j) = MAP;
end
fprintf('============================================%d bits SMH-OQA mAP over %d iterations:%.4f=============================================\n', param.bits, run, mean(map));
