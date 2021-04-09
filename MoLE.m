% Main classe of MoLE-EN
% For use with MATLAB
% Copyright (C) 2021 -- Francisco Souza <alexandre.andry@gmail.com> or <f.souza@science.ru.nl>
% MoLE-EN comes with ABSOLUTELY NO WARRANTY;
% In case of application of this method in any publication,
% please, cite the original work:
% Souza, Francisco; Mendes, Jérôme; Araújo, Rui. 2021. 
% "A Regularized Mixture of Linear Experts for Quality Prediction in Multimode and Multiphase Industrial Processes" 
% Appl. Sci. 11, no. 5: 2040. 
% DOI: https://doi.org/10.3390/app1105204

classdef MoLE
    properties
        ne = 2;
        reg = 'lasso';
        lambda = 1e-3;
        iterations = 50;
        theta
        v
    end
    methods
        function obj = MoLE(ne,iterations,reg,lambda)
            if nargin > 0
                obj.ne = ne;
            end
            if nargin > 1
                obj.iterations = iterations;
            end
            if nargin > 2
                obj.reg = reg;
            end
            if nargin > 3
                obj.lambda = lambda;
            end
        end
        
        function obj = fit(obj,x,y)
            [m,n]=size(x);
            
            % obj initialization
            ne_ = obj.ne;
            n_int = obj.iterations;
            
            % initialization
            tol = 1e-3;
            bic_min = 1e10;
            improvement = 0;
            theta_ = randn(n+1,obj.ne)*.1; % expert parameter
            v_ = randn(n+1,obj.ne)*.1; % gate parameter
            v_(:,1)=0; % gate parameter
            
            prob = zeros(m,obj.ne); %
            resp = zeros(m,obj.ne); %
            gate = zeros(m,obj.ne); %
            gv = 100*ones(1,obj.ne);
            ypred = zeros(m,obj.ne);
            e = zeros(m,obj.ne);
            Q_plot = zeros(1,m);
            
            for i=1:n_int
                
                for j=1:ne_ % compute the error of each model
                    ypred(:,j) = [ones(m,1) x]*theta_(:,j);
                    prob(:,j) = normpdf(y,ypred(:,j),sqrt(gv(j)) + tol);
                    e(:,j) = (y - ypred(:,j)).^2;
                end
                
                for j=1:ne_
                    gate(:,j) = [ones(m,1) x]*v_(:,j);
                end
                pi = exp(gate - repmat(logsumexp(gate),1,ne_));
                
                
                for l=1:m
                    tmp=prob(l,:)*pi(l,:)';
                    for j=1:ne_ % compute weights
                        resp(l,j) = (pi(l,j)*prob(l,j)+eps)/(tmp+eps);
                    end
                end
                
                % calcula likelihood function
                Q  =0;
                for l=1:m
                    for j=1:ne_ % compute weights
                        Q = Q + resp(l,j)*(log(pi(l,j)+eps) + log(prob(l,j)+eps));
                    end
                end
                Q_plot(i) = Q;
                
                % M STEP
                % update parameters
                % expert function update
                for j=1:ne_
                    w =  resp(:,j);
                    theta_(:,j) = obj.fit_expert_model([ones(m,1) x],y,w,theta_(:,j));
                end
                
                % gate function update
                
                df_hmole = 0;
                for j=2:ne_
                    
                    for l=1:ne_
                        gate(:,l) = [ones(m,1) x]*v_(:,l);
                    end
                    pi = exp(gate - repmat(logsumexp(gate),1,ne_));
                    
                    w=(pi(:,j).*(1-pi(:,j)));
                    for rep=1:30
                        y_gate=[ones(m,1) x]*v_(:,j) + (w.*((w.^2+1e-3).^-1)).*(resp(:,j)-pi(:,j));
                        
                        % fit lower gates
                        [v_tmp,df] = obj.fit_gate_model([ones(m,1) x],y_gate,w,v_(:,j));
                        tmp = v_(:,j);
                        v_(:,j) = v_tmp;
                        if convergenceTest(v_(:,j), tmp, 1e-3)
                            break;
                        end
                        
                    end
                    df_hmole = df_hmole + df;
                end
                
                for j=1:ne_
                    ypred(:,j) = [ones(m,1) x]*theta_(:,j);
                    e(:,j) = (y - ypred(:,j)).^2;
                    E = sum(resp(:,j).*e(:,j));
                    gv(j)=((E+eps)/sum(resp(:,j))+eps);
                end
                
                y_est_train = sum(ypred.*pi,2);
                bic = m*log(sum((y - y_est_train).^2)/m) + log(m)*df_hmole;
                rss = sum((y - y_est_train).^2);
                %[rss bic]
                
                if bic<bic_min
                    bic_min = bic;
                    improvement = 0;
                    obj.v = v_;
                    obj.theta = theta_;
                else
                    improvement =improvement +1;
                end
                
                if i>50 && improvement>30
                    break;
                end
                
            end
        end
        function [y_est_mole,gates] = predict(obj,x)
            for j=1:obj.ne
                gate_out(:,j) = [ones(size(x,1),1) x]*obj.v(:,j);
                ypred(:,j) = [ones(size(x,1),1) x]*obj.theta(:,j);
            end
            pi = exp(gate_out - repmat(logsumexp(gate_out),1,obj.ne));
            
            y_est_mole = sum(ypred.*pi,2);
            gates = pi;
            
        end
        
        function [v,df] = fit_gate_model(obj,x,y,w,theta)
            [v,df] = fit_mlr(x,y,w,theta,obj.reg,obj.lambda);
        end
        
        function [theta,df] = fit_expert_model(obj,x,y,w,theta)
            [theta,df] = fit_mlr(x,y,w,theta,obj.reg,obj.lambda);
        end
        
    end
end


function [theta,df] = fit_mlr(x,y,w,theta,reg,lambda)
% mlr
% options.expert_solver = {'mlr','mlr-cgd','mlr-lasso','mlr-en','mlr-rr','pls'}
% options.expert_quality_of_fit:  {'bic','calibration','kfold'}

if strcmp(reg,'mlr')
    theta = lscov(x,y,w);
    df = size(x,2);
elseif strcmp(reg,'lasso') || strcmp(reg,'en') || ...
        strcmp(reg,'rr')
    if strcmp(reg,'rr')
        alpha = 0;
    elseif strcmp(reg,'lasso')
        alpha=1;
    elseif strcmp(reg,'en')
        alpha=0.5;
    end
    
    den=sum(repmat(w,1,size(x,2)).*x.^2,1);
    for jj=1:size(x,2)
        tmp = sum(w.*x(:,jj).*(y-x(:,[1:jj-1 jj+1:end])*theta([1:jj-1 jj+1:end])));
        theta(jj) = softthrh(tmp,lambda*alpha)/(eps+den(jj)+lambda*(1-alpha)); % en penalty
    end
    df = sum(theta~=0);
end

end

function [converged] = convergenceTest(fval, previous_fval, threshold, warn)
% Check if an objective function has converged
%
% We have converged if the slope of the function falls below 'threshold',
% i.e., |f(t) - f(t-1)| / avg < threshold,
% where avg = (|f(t)| + |f(t-1)|)/2
% 'threshold' defaults to 1e-4.
% This stopping criterion is from Numerical Recipes in C p423
% This file is from pmtk3.googlecode.com


if nargin < 3, threshold = 1e-3; end
if nargin < 4, warn = false; end

converged = 0;
delta_fval = abs(fval - previous_fval);
avg_fval = (abs(fval) + abs(previous_fval) + eps)/2;
if (delta_fval / avg_fval) < threshold
    converged = 1;
end

if warn && (fval-previous_fval) < -2*eps %fval < previous_fval
    warning('convergenceTest:fvalDecrease', 'objective decreased!');
end

%  converged = 0;
end

function s=logsumexp(x)
y=max(x,[],2);
s=y+log(sum(exp(x-repmat(y,1,size(x,2))),2));
end
