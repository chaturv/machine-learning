function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
% fprintf('X: \n');
% fprintf(' rows: %f \n', size(X, 1));
% fprintf(' cols: %f \n', size(X, 2));

X = [ones(m, 1) X];

% fprintf('X: \n');
% fprintf(' rows: %f \n', size(X, 1));
% fprintf(' cols: %f \n', size(X, 2));

A1 = transpose(X);

Zeta2 = Theta1 * A1;
A2 = sigmoid(Zeta2);

% Add ones to the A2 data matrix
A2 = [ones(1, size(A2, 2)); A2];

% fprintf('A2: \n');
% fprintf(' rows: %f \n', size(A2, 1));
% fprintf(' cols: %f \n', size(A2, 2));

Zeta3 = Theta2 * A2;
A3 = sigmoid(Zeta3);

% transpose A3 and extract max and index
[MAX_VAL, I] = max(transpose(A3), [], 2);

p = I;

% fprintf('p: \n');
% fprintf(' rows: %f \n', size(p, 1));
% fprintf(' cols: %f \n', size(p, 2));

% =========================================================================

end
