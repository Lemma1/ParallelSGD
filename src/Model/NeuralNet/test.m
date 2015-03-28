%% Load Data
%  To load different dataset, just uncomment the corresponding part. Also, 
%  feel free the change the network architexture by chaning the line
%  "num_neuron = [D, 4, 1]". Here, the ith element in the "num_neuron" 
%  array means the number of neurons used for the ith layer 

clear; clc;
DEBUG = 0;

%
X = [1, 3; 
     2, 4];
y = [1, 0; 
     0, 1];

[D, N] = size(X);
C = size(y, 1);

% network architecture
num_neuron = [D, 3, 3, 2];
num_layer  = numel(num_neuron);
num_weight = num_layer - 1;


% activation function
sigmoid = @(x) 1 ./ (1 + exp(-x));

% weights initialization
W     = cell(num_weight,1);
gradW = cell(num_weight,1);
for layer = 1:num_weight
    fan_in  = num_neuron(layer)+1;
    fan_out = num_neuron(layer+1);
    multiplier = 4 * sqrt(6 / (fan_in + fan_out));
    W{layer} = ones(fan_in, fan_out);
    
%     W{layer} = multiplier * (2 * rand(fan_in, fan_out) - 1);
    gradW{layer} = zeros(size(W{layer}));    
end

% training parameters
activation = cell(num_layer,1);
delta      = cell(num_layer,1);

% iterations  
num_trials = 1;
for t = 1:num_trials
    % forward pass
    % input-to-hidden
    activation{1} = X;
    % hidden-to-hidden&hidden-to-output
    for layer = 2:num_layer
        activation{layer} = sigmoid(W{layer-1}' * [activation{layer-1}; ones(1, N)]);
    end
    
    p = exp(activation{num_layer}) ./ repmat(sum(exp(activation{num_layer})), C, 1);
    error = y .* log(p);
    error = sum(error(:))

    % backward pass
    % output-to-hidden
    errorsig = p - y
    dsigmoid = activation{num_layer} .* (1 - activation{num_layer})
    delta{num_layer} = dsigmoid .* errorsig;
    gradW{num_layer-1} = [activation{num_layer-1}; ones(1, N)] * delta{num_layer}';
    
    % hidden-to-hidden&hidden-to-input
    for layer = num_layer-1:-1:2
        errorsig = W{layer} * delta{layer+1};
        dsigmoid = activation{layer} .* (1 - activation{layer});
        delta{layer} = dsigmoid .* errorsig(1:end-1,:);
        gradW{layer-1} = [activation{layer-1}; ones(1, N)] * delta{layer}';        
    end
end