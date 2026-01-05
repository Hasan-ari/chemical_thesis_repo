classdef SimpleRNNLayer < nnet.layer.Layer & nnet.layer.Formattable
    % SimpleRNNLayer   A custom vanilla RNN layer for sequence modelling
    %
    %   layer = SimpleRNNLayer(numHiddenUnits, name) creates a simple RNN
    %   layer with the specified number of hidden units and optional name.

    properties
        NumHiddenUnits
    end

    properties (Learnable)
        % Input-to-hidden weights
        WeightsInput
        % Hidden-to-hidden weights
        WeightsHidden
        % Bias
        Bias
    end

    methods
        function layer = SimpleRNNLayer(numHiddenUnits, name)
            % Constructor
            if nargin < 2
                name = "";
            end
            layer.Name = name;
            layer.Description = "Simple RNN layer with " + numHiddenUnits + " hidden units";
            layer.NumHiddenUnits = numHiddenUnits;

            % Xavier/Glorot initialization
            szInput = [numHiddenUnits 1]; % will be resized on first forward pass
            layer.WeightsInput = randn(numHiddenUnits, 1) * sqrt(2/(numHiddenUnits+1));
            layer.WeightsHidden = randn(numHiddenUnits, numHiddenUnits) * sqrt(2/(2*numHiddenUnits));
            layer.Bias = zeros(numHiddenUnits, 1);
        end

        function Z = predict(layer, X)
            % X is [features x time x batch]
            [numFeatures, numTimeSteps, batchSize] = size(X);

            % If first call, init input weights properly
            if size(layer.WeightsInput,2) ~= numFeatures
                layer.WeightsInput = randn(layer.NumHiddenUnits, numFeatures) * sqrt(2/(numFeatures+layer.NumHiddenUnits));
            end

            H = zeros(layer.NumHiddenUnits, batchSize, 'like', X);

            for t = 1:numTimeSteps
                Xt = X(:,t,:);
                Xt = reshape(Xt, numFeatures, batchSize);
                H = tanh(layer.WeightsInput * Xt + layer.WeightsHidden * H + layer.Bias);
            end

            % Output last hidden state
            Z = H;
        end

        function [Z, memory] = forward(layer, X)
            % Forward pass with memory for backprop
            [numFeatures, numTimeSteps, batchSize] = size(X);

            if size(layer.WeightsInput,2) ~= numFeatures
                layer.WeightsInput = randn(layer.NumHiddenUnits, numFeatures) * sqrt(2/(numFeatures+layer.NumHiddenUnits));
            end

            H = zeros(layer.NumHiddenUnits, batchSize, 'like', X);
            memory.H = cell(1, numTimeSteps);
            memory.X = X;

            for t = 1:numTimeSteps
                Xt = X(:,t,:);
                Xt = reshape(Xt, numFeatures, batchSize);
                H = tanh(layer.WeightsInput * Xt + layer.WeightsHidden * H + layer.Bias);
                memory.H{t} = H;
            end

            Z = H;
        end

        function [dX, dW_in, dW_hid, dB] = backward(layer, X, ~, dZ, memory)
            % Backpropagation through time (BPTT)
            [numFeatures, numTimeSteps, batchSize] = size(X);

            dW_in = zeros(size(layer.WeightsInput), 'like', X);
            dW_hid = zeros(size(layer.WeightsHidden), 'like', X);
            dB = zeros(size(layer.Bias), 'like', X);
            dX = zeros(size(X), 'like', X);

            dH_next = dZ;

            for t = numTimeSteps:-1:1
                H_t = memory.H{t};
                if t > 1
                    H_prev = memory.H{t-1};
                else
                    H_prev = zeros(size(H_t), 'like', X);
                end

                Xt = X(:,t,:);
                Xt = reshape(Xt, numFeatures, batchSize);

                % Derivative of tanh
                dtanh = (1 - H_t.^2) .* dH_next;

                dW_in = dW_in + dtanh * Xt';
                dW_hid = dW_hid + dtanh * H_prev';
                dB = dB + sum(dtanh, 2);

                dX(:,t,:) = reshape(layer.WeightsInput' * dtanh, numFeatures, 1, batchSize);
                dH_next = layer.WeightsHidden' * dtanh;
            end
        end
    end
end