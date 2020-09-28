function [model, B_te,B_db] = solve_UMHOQA(X, Y,I_te,T_te,I_db,T_db,model)
% X           the 1st modality
% Y           the 2nd modality
% H_x         the latent representation for 1st modality
% H_y         the latent representation for 2st modality
% Rx          the latent space basis for 1st modality
% Ry          the latent space basis for 2st modality
% B           hash code
tic
obj = Inf;
X = single(X');
sampleMean = mean(X,1);
X = (X - repmat(sampleMean,size(X,1),1));
Y = single(Y');
sampleMean = mean(Y,1);
Y = (Y - repmat(sampleMean,size(Y,1),1));
[px, nx] = size(X);
[py, ny] = size(Y);
d = model.d;
m = model.m;
k = model.k;
alpha = model.alpha;
if length(k) == 1
    k = ones(m, 1) * k;
end
bits = sum(log2(k));
model.bits = bits;

if ~exist('iters', 'var') || isempty(iters)
    iters = 30;
end
model.iters = iters;

%% matrix initialization
Rx = eye(px, d, 'single');
Ry = eye(py, d, 'single');
RxX = Rx' * X;
RyY = Ry' * Y;
Cx = cell(1, m);
Cy = cell(1, m);
for i = 1:m
    perm = randperm(nx, k(i));
    Cx{i} = RxX(:, perm);
    perm = randperm(ny, k(i));
    Cy{i} = RyY(:, perm);
end
B =randi(15,nx, m,'double');
CxB = zeros([d, nx], 'single');
CyB = zeros([d, ny], 'single');
p1=0.5;
p2=0.5;
H_x = (alpha*(Rx'*Rx))\( alpha* Rx' * X + (1/p1) * CxB);
H_y = (alpha*(Ry'*Ry))\( alpha* Ry' * Y + (1/p2) * CyB);
%% Iterative algorithm
for iter = 0:iters
    if (mod(iter, 1) == 0)
        objlast = obj;
        obj =alpha*  mean(mean((X - Rx * H_x).^2)) + alpha*mean(mean((Y - Ry * H_y).^2))+ ...
            (1/p1)*mean(mean((H_x-CxB).^2))+ (1/p2)*mean(mean((H_y-CyB).^2));
        fprintf('%3d  %f\n', iter, obj);
        model.obj(iter + 1) = obj;
    end
    if objlast - obj < model.obj(1) * 1e-10
        fprintf('algorithm converged!\n')
        break;
    end
    
    %update p
    p1 = double(sqrt(sum(sum((H_x - CxB).^2)))) / double((sqrt(sum(sum((H_x - CxB).^2)))+sqrt(sum(sum((H_y - CyB).^2)))));
    p2 = double(sqrt(sum(sum((H_y - CyB).^2)))) / double((sqrt(sum(sum((H_x - CxB).^2)))+sqrt(sum(sum((H_y - CyB).^2)))));    
    %update R
    [Ux, ~, Vx] = svd(X * H_x', 0);
    [Uy, ~, Vy] = svd(Y * H_y', 0);
    Rx = Ux * Vx';
    Ry = Uy * Vy';   
    %update C
    eyek = speye(k(1));
    Bx2 = [];
    for i = 1:m
        Bx2 = [Bx2, eyek(B(:, i), :)];
    end
    C2 = ((1/p1) * double(H_x) * Bx2 + (1/p2) * double(H_y) * Bx2) / (((1/p1) + (1/p2)) * (Bx2' * Bx2)+ 1e5*eye(sum(k))) ;
    C0 = mat2cell(single(C2), d, k);
    Cx = C0;
    Cy = C0;   
    %update H
    H_x = (alpha*(Rx'*Rx))\( alpha* Rx'*X + (1/p1)*CxB);
    H_y = (alpha*(Ry'*Ry))\( alpha* Ry'*Y + (1/p2)*CyB);    
    %update B
    eRxX = H_x;
    eRyY = H_y;
    CxB = zeros([d, nx], 'single');
    CyB = zeros([d, ny], 'single');
    for i = 1:m
        B0 = icm(Cx{i}, eRxX(:, 1:nx), Cy{i}, eRyY(:, 1:nx), (1/p1), (1/p2));
        B(1:nx, i) = B0;
        eRxX = eRxX - Cx{i}(:, B(:, i));
        CxB = CxB  + Cx{i}(:, B(:, i));
        eRyY = eRyY - Cy{i}(:, B(:, i));
        CyB = CyB  + Cy{i}(:, B(:, i));
    end
end
for i = 1:m
    model.Cx{i} = Cx{i};
    model.Cy{i} = Cy{i};
end
toc
%% Online stage
%query
iters = 30;
X = single(I_te');
Y = single(T_te');
nn = size(X,2);
Rx = eye(px, d, 'single');
Ry = eye(py, d, 'single');
Cx = model.Cx;
Cy = model.Cy;
B = zeros(nn, m, 'int32');
CxB = zeros([d, nn], 'single');
CyB = zeros([d, nn], 'single');
H_x = (alpha*(Rx'*Rx))\( alpha* Rx'*X + (1/p1)*CxB);
H_y = (alpha*(Ry'*Ry))\( alpha* Ry'*Y + (1/p2)*CyB);
for iter = 0:iters
    %update  query R
    [Ux, ~, Vx] = svd(X * H_x', 0);
    [Uy, ~, Vy] = svd(Y * H_y', 0);
    Rx = Ux * Vx';
    Ry = Uy * Vy';
    %update query H
    H_x = (alpha*(Rx'*Rx))\( alpha* Rx' * X + (1/p1) * CxB);
    H_y = (alpha*(Ry'*Ry))\( alpha* Ry' * Y + (1/p2) * CyB);
    %update query p
    p1 = double(sqrt(sum(sum((H_x - CxB).^2)))) / double((sqrt(sum(sum((H_x - CxB).^2)))+sqrt(sum(sum((H_y - CyB).^2)))));
    p2 = double(sqrt(sum(sum((H_y - CyB).^2)))) / double((sqrt(sum(sum((H_x - CxB).^2)))+sqrt(sum(sum((H_y - CyB).^2)))));
    %update query B
    eRxX = H_x;
    eRyY = H_y;
    CxB = zeros([d, nn], 'single');
    CyB = zeros([d, nn], 'single');
    for i = 1:model.m
        B0 = icm(Cx{i}, eRxX(:, 1:nn), Cy{i}, eRyY(:, 1:nn), (1/p1), (1/p2));
        B(1:nn, i) = B0;
        eRxX = eRxX - Cx{i}(:, B(:, i));
        CxB = CxB + Cx{i}(:, B(:, i));
        eRyY = eRyY - Cy{i}(:, B(:, i));
        CyB = CyB + Cy{i}(:, B(:, i));
    end
    if (mod(iter, 1) == 0)
        objlast = obj;
        obj =  alpha*mean(mean((X - Rx * H_x).^2)) + alpha*mean(mean((Y - Ry * H_y).^2))+ (1/p1)*mean(mean((H_x-CxB).^2))+ (1/p2)*mean(mean((H_y-CyB).^2));
        fprintf('%3d  %f\n', iter, obj);
        model.obj(iter + 1) = obj;
    end
    
    if iter > 3 &&  objlast - obj < model.obj(1) * 1e-3
        fprintf('algorithm converged!\n')
        break;
    end
    
end
B_te = B-1;
% Database
iters = 30;
X = single(I_db');
Y = single(T_db');
nn = size(X,2);
Rx = eye(px, d, 'single');
Ry = eye(py, d, 'single');
Cx = model.Cx;
Cy = model.Cy;
B = zeros(nn, m, 'int32');
CxB = zeros([d, nn], 'single');
CyB = zeros([d, nn], 'single');
H_x = (alpha*(Rx'*Rx))\( alpha* Rx'*X + (1/p1)*CxB);
H_y = (alpha*(Ry'*Ry))\( alpha* Ry'*Y + (1/p2)*CyB);
for iter = 0:iters
    %update  datebase R
    [Ux, ~, Vx] = svd(X * H_x', 0);
    [Uy, ~, Vy] = svd(Y * H_y', 0);
    Rx = Ux * Vx';
    Ry = Uy * Vy';
    %update  datebase H
    H_x = (alpha*(Rx'*Rx))\( alpha* Rx' * X + (1/p1) * CxB);
    H_y = (alpha*(Ry'*Ry))\( alpha* Ry' * Y + (1/p2) * CyB);
    %update  datebase p
    p1 = double(sqrt(sum(sum((H_x - CxB).^2)))) / double((sqrt(sum(sum((H_x - CxB).^2)))+sqrt(sum(sum((H_y - CyB).^2)))));
    p2 = double(sqrt(sum(sum((H_y - CyB).^2)))) / double((sqrt(sum(sum((H_x - CxB).^2)))+sqrt(sum(sum((H_y - CyB).^2)))));
    %update  datebase B
    eRxX = H_x;
    eRyY = H_y;
    CxB = zeros([d, nn], 'single');
    CyB = zeros([d, nn], 'single');
    for i = 1:model.m
        B0 = icm(Cx{i}, eRxX(:, 1:nn), Cy{i}, eRyY(:, 1:nn), (1/p1), (1/p2));
        B(1:nn, i) = B0;
        eRxX = eRxX - Cx{i}(:, B(:, i));
        CxB = CxB + Cx{i}(:, B(:, i));
        eRyY = eRyY - Cy{i}(:, B(:, i));
        CyB = CyB + Cy{i}(:, B(:, i));
    end
    if (mod(iter, 1) == 0)
        objlast = obj;
        obj = alpha * mean(mean((X - Rx * H_x).^2)) + alpha* mean(mean((Y - Ry * H_y).^2))+ (1/p1)*mean(mean((H_x-CxB).^2))+ (1/p2)*mean(mean((H_y-CyB).^2));
        fprintf('%3d  %f\n', iter, obj);
        model.obj(iter + 1) = obj;
    end
    if iter > 3 && objlast - obj < model.obj(1) * 1e-3
        fprintf('algorithm converged!\n')
        break;
    end
    
end
B_db = B-1;