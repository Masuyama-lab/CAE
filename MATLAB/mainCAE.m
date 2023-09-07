% 
% Copyright (c) 2022 Naoki Masuyama (masuyama@omu.ac.jp)
% This software is released under the MIT License.
% http://opensource.org/licenses/mit-license.php
% 

clear all

rng(1);

% Dataset
data_name = "Iris";

% Load data
tmpData = load(strcat(data_name,'.mat'));
data = tmpData.data ;
labels = tmpData.target;

if min(labels)==0
    labels = labels+1;
end

% Randamization
ran = randperm(size(data,1));
data = data(ran,:);
labels = labels(ran);
    


% Parameters of CAE =======================================================
net.numNodes    = 0;   % the number of nodes
net.weight      = [];  % node position
net.CountNode = [];    % winner counter for each node
net.edge = [];         % Initial connections (edges) matrix
net.adaptiveSig = [];  % kernel bandwidth for CIM in each node
net.LabelCluster = []; % Cluster label for connected nodes
net.V_thres_ = []; % similarlity thresholds
net.activeNodeIdx = [];% nodes for SigmaEstimation
net.CountLabel = [];   % counter for labels of each node
net.numSample = 0;     % number of samples 
net.flag_set_lambda = false;  % a flag for setting lambda
net.numActiveNode = size(data,1); % number of active nodes
net.divMat(1,1) = 1;   % a matrix for diversity via determinants
net.div_lambda = inf; % \lambda determined by diversity via determinants
net.lifetime_d_edge = 0; % average lifetime of deleted edges
net.n_deleted_edge = 0; % number of deleted edges
net.sigma = 0;  % an estimated sigma for CIM
% =========================================================================


time = 0;

% Training
tic
net = CAE_train(data, net);
time = time + toc;

% Test
predicted_labels = CAE_test(data, net);

% Evaluation
[NMI, AMI, ARI] = Evaluate_Clustering_Performance(labels, predicted_labels);

% Results
disp(['   # of nodes: ', num2str(net.numNodes)])
disp(['# of clusters: ', num2str(max(net.LabelCluster))])
disp(['          ARI: ', num2str(ARI)])
disp(['          AMI: ', num2str(AMI)])
disp(['          NMI: ', num2str(NMI)])
disp(['   Train Time: ', num2str(time)])
disp(' ')
