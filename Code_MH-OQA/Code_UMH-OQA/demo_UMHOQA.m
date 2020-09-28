clear all;
warning off;
clc;

%% Dataset Loading
load('D:\Hashing produce\datasets\mir_cnn.mat')
fprintf('MIR Flickr_CNN dataset loaded...\n');

%% Parameters Setting
run = 5;
model.MAP=[];
bits = 32;
bookbits = 4;
m = bits / bookbits;
k = 2 ^ bookbits;
model.m = m;
model.k = k;
model.bookbits = bookbits;
d = 1000;
alpha = 100;
model.d = d;
model.alpha = alpha;

%% Training & Evaluation Process
fprintf('\n============================================Start training UMH-OQA============================================\n');
for j = 1:run
    % Training model 
    [model,B_te,B_db] = solve_UMHOQA(I_tr, T_tr, I_te,T_te,I_db,T_db,model);   
    % Evaluation  
    Dhamm=hammingDist(B_db,B_te);
    [MAP] = perf_metric4Label(L_db, L_te,Dhamm);
    map(j) = MAP;
end
fprintf('============================================%d bits UMH-OQA mAP over %d iterations:%.4f=============================================\n', param.bits, run, mean(map));
