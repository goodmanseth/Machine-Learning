function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% remember that the regularization only affects theta(2:end)

h = sigmoid(X*theta);

main_cost = -(1/m)*sum(y.*log(h)+(1-y).*log(1-h)); % same as without regularization
lambda_cost = lambda/(2*m)*sum(theta(2:end).*theta(2:end)); 

J = main_cost + lambda_cost;

main_grad = (1/m)*(X'*(h-y)); % same as without regularization
lambda_grad = (lambda/m)*theta;
lambda_grad(1) = 0;

grad = main_grad + lambda_grad;



% =============================================================

end
