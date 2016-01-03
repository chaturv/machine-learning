function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% fprintf('lambda = %f \n', lambda);

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% A1 as reference variable input layer 
A1 = X;

% y_matrix of dim 5000x10 created from y
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);


DELTA1 = zeros(size(Theta1));
DELTA2 = zeros(size(Theta2));

% for all training examples
for i = 1:m
    A1_i = transpose(A1(i,:));    % each example (400 x 1; all pixel values)
    A1_i = [ ones(1); A1_i ];     % add bias unit
    
    zeta2_i = Theta1 * A1_i;
    A2_i = sigmoid(zeta2_i);
    A2_i = [ ones(1); A2_i ];     % add bias unit
    
    zeta3_i = Theta2 * A2_i;
    A3_i = sigmoid(zeta3_i);    
    
    % hx as reference variable for A3_i
    hx = A3_i;           
    
    yk = transpose(y_matrix(i,:));
      
    % ALTERNATE APPROACH: init y vector of k elements (one vs all)
    
    % yk = zeros(num_labels, 1);
    % yi = y(i);       % get the i-th training example y value   
    % set value
    % yk(yi) = 1;    
    
    % cost function
    J = J + sum( yk .* log(hx) + (1 - yk) .* log(1 - hx) );   
    
    % find delta_3: error of the output layer. dim 10 x 1
    delta_3 = hx - yk;
    
    % find delta_2: error of the hidden layer. dim 26 x 1
    delta_2 = (transpose(Theta2) * delta_3) .* [ones(1); sigmoidGradient(zeta2_i)];
    % get rid of delta(2)(0). dim 25 x 1
    delta_2 = delta_2(2:end);
    
    % accumulate gradients for layer 2
    DELTA2 = DELTA2 + delta_3 * transpose(A2_i);    

    % accumulate gradients for layer 1
    DELTA1 = DELTA1 + delta_2 * transpose(A1_i);    
    
end

% cost function regularization parameters
Theta1_reg = sum(sum(Theta1(:,2:end) .^ 2));
Theta2_reg = sum(sum(Theta2(:,2:end) .^ 2));

J = (-1 / m) * J + (lambda / (2 * m)) * (Theta1_reg + Theta2_reg); 
% -------------------------------------------------------------

% gradients
% add reg terms for DELTA2 and divide by m
Theta2_grad = (DELTA2 + lambda * [zeros(size(Theta2,1), 1) Theta2(:,2:end)]) ./ m;
% add reg terms for DELTA1 and divide by m
Theta1_grad = (DELTA1 + lambda * [zeros(size(Theta1,1), 1) Theta1(:,2:end)]) ./ m;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
