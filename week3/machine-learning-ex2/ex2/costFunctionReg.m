function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2); % number of features (including the default x0 = 1)


% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% hypothesis function
hx = sigmoid(X * theta);

% origial cost function
J = (-1 / m) * sum(y .* log(hx) + (1 - y) .* log(1 - hx)) + (lambda / (2 * m)) * sum(theta(2:end) .^ 2); 

% gradient
%grad(1) = (1 / m) * sum(hx - y);

%for i=2:n
%    grad(i) = (1 / m) * (sum((hx - y) .* X(:,i)) + lambda * theta(i));
%end    

grad = (1/m) * (sum(repmat((hx - y), 1, n) .* X) + transpose(theta) * lambda);
% replace 
grad(1) = (1 / m) * sum(hx - y);

grad = grad(:);

% =============================================================

end
