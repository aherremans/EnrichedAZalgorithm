%% Poisson equation in a rectangular domain
% "Enriched spectral methods and applications to problems with weakly
% singular solutions" [Chen, Shen 2018] --> section 3.1
% ! this script requires chebfun
addpath('../util');

sqrtNlist = 4:2:16;
f = @(x,y) exp(x+y);

errors0 = zeros(length(sqrtNlist)-1,1);                 % errors of standard Galerkin method
errorsESG = zeros(length(sqrtNlist)-1,1);               % errors of ESG-II
errorsAZ = zeros(length(sqrtNlist)-1,1);                % errors of Galerkin + collocation ('AZ')
MK = 2;                                                 % # smoothness constraints
grd = linspace(-1,1,7); grd = grd(2:end-1);             % grid for collocation constraints
[xx,yy] = ndgrid(grd,grd); xx = xx(:); yy = yy(:);    

% error grid
grd = linspace(-1,1,100); grd = grd(2:end-1);
[xx_error,yy_error] = ndgrid(grd,grd);
xx_error = xx_error(:); yy_error = yy_error(:);
Merror = length(xx_error);

% reference solution is ESG-II with N = 30
load('reference_sol.mat');

% use recurrence relations to construct Galerkin system matrices
% "Efficient spectral-Galerkin algorithms for direct solution for
% second-order differential equations using Jacobi polynomials" [Doha,
% Bhrawy 2006] --> corollary 3.3
xn = @(n) 1./(n+1);
hn = @(n) 2^3*(n+1)./((2*n+3).*(n+2));

% construct singular function
% [Chen, Shen 2018] --> section 3.1
r = @(x,y) sqrt((1-x).^2 + (1-y).^2);
theta = @(x,y) atan((1-x)./(1-y));
s = @(x,y) 1/2 - (1/pi)*r(x,y).^2.*(log(r(x,y)).*sin(2*theta(x,y)) + ...
    theta(x,y).*cos(2*theta(x,y))) - r(x,y).*cos(theta(x,y)) + ...
    r(x,y).^2.*cos(2*theta(x,y))/2;
S = @(x,y) f(1,1)*s(x,y) + f(-1,1)*s(-x,y) + f(1,-1)*s(x,-y) + f(-1,-1)*s(-x,-y);

% homogenize the singular function
% "Efficient spectral-Galerkin method I. direct solvers of second- and
% fourth-order equations using Legendre polynomials" [Shen 1994]
% --> subsection 4.2
% ! chebfun2 has problems with NaNs --> avoid this by perturbing domain with eps
u1 = @(x,y) (S(x,1-eps) - S(x,-1+eps)).*y/2 + (S(x,1-eps) + S(x,-1+eps))/2;
u2 = @(x,y) ((S(1-eps,y) - u1(1-eps,y)) - (S(-1+eps,y) - u1(-1+eps,y))).*x/2 + ...
    ((S(1-eps,y) - u1(1-eps,y)) + (S(-1+eps,y) - u1(-1+eps,y)))/2;
domain = [-1+eps 1-eps -1+eps 1-eps];
psi = chebfun2( @(x,y) S(x,y) - u1(x,y) - u2(x,y), domain);
psi_dx = diff(psi,1,2);
psi_dy = diff(psi,1,1);
Lpsi = -laplacian(psi); 

for k = 1:length(sqrtNlist)
    sqrtN = sqrtNlist(k)

    % Galerkin system
    Cinv = diag(((1:(sqrtN-1)).*(2:sqrtN)).^(-1));
    B = zeros(sqrtN-1,sqrtN-1);
    for n = 1:(sqrtN-1)
        B(n,n) = 2*n*(n+1)/((2*(n-1)+1)*(2*(n-1)+5));
        if(n > 2)
            B(n,n-2) = -n*(n+1)/((2*(n-1)-1)*(2*(n-1)+1));
        end
        if(n < sqrtN-2)
            B(n,n+2) = -n*(n+1)/((2*(n-1)+5)*(2*(n-1)+7));
        end
    end

    % compute right-hand sides
    F = zeros(sqrtN-1,sqrtN-1);
    Fpsi = zeros(sqrtN-1,sqrtN-1);
    for i = 0:(sqrtN-2)
        for j = 0:(sqrtN-2)
            F(i+1,j+1) = integral2(@(x,y) f(x,y) .* phi(i,j,x,y), -1, 1, -1, 1) ...
                /((xn(i)^2*hn(i))*(xn(j)^2*hn(j))); 
            Fpsi(i+1,j+1) = integral2(@(x,y) psi_dx(x,y) .* phi_dx(i,j,x,y) ...
                + psi_dy(x,y) .* phi_dy(i,j,x,y), -1, 1, -1, 1, ...
                'RelTol', 1e-8)/((xn(i)^2*hn(i))*(xn(j)^2*hn(j))); 
            % (evaluating these integrals is the main bottleneck of this
            % script, optimizations consist of smart quadrature rules (see
            % also remark 2.1 in [Chen, Shen 2018]) and searching for an
            % explicit expression for the derivatives and laplacian of the
            % singular function psi)
        end
    end
    
    % solution using classical spectral-Galerkin (i.e. only with smooth
    % basis)
    U = sylvester(Cinv*B, transpose(B)*Cinv, Cinv*F*Cinv);
    sol0 = @(x,y) evalsmooth(U,x,y);
    errors0(k) = norm(sol0(xx_error,yy_error) - solution, 2)/sqrt(Merror);

    % new singular basis = singular functions - their approximation in the
    % smooth basis
    psi_j = sylvester(Cinv*B, transpose(B)*Cinv, Cinv*Fpsi*Cinv);
    smoothpsi = @(x,y) evalsmooth(psi_j,x,y);
    newpsi = @(x,y) psi(x,y) - smoothpsi(x,y);

    % solution using ESG-II (approximation in new singular basis using
    % smoothness condition)
    psi_j_K = psi_j(end-MK+1:end,end-MK+1:end); U_K = U(end-MK+1:end,end-MK+1:end);
    s = psi_j_K(:) \ U_K(:);
    solESG = @(x,y) evalsmooth(U,x,y) + s*newpsi(x,y);
    errorsESG(k) = norm(solESG(xx_error,yy_error) - solution, 2)/sqrt(Merror);

    % solution using AZ algorithm (approximation in new singular  
    % basis using collocation constraints)
    % (can be optimized by writing out laplacian of smooth basis functions
    % explicitly)
    Lsol0 = -laplacian(chebfun2(@(x,y) sol0(x,y), domain));
    Lnewpsi = Lpsi + laplacian(chebfun2(@(x,y) smoothpsi(x,y), domain));
    sAZ = Lnewpsi(xx,yy) \ (f(xx,yy) - Lsol0(xx,yy));
    solAZ = @(x,y) evalsmooth(U,x,y) + sAZ*newpsi(x,y);
    errorsAZ(k) = norm(solAZ(xx_error,yy_error) - solution, 2)/sqrt(Merror);
end

% convergence plot
figure;
semilogy(sqrtNlist.^2,errors0, '.-'); xlabel('N'); ylabel('L2 error (compared to reference)');
hold on; semilogy(sqrtNlist.^2,errorsESG, '.-'); semilogy(sqrtNlist.^2,errorsAZ, '.-'); 
legend('classical SG', 'ESG-II', 'AZ');

%% auxiliary functions to evaluate smooth basis functions
function sol = phi(i,j,x,y)
    xx = x(:); yy = y(:);
    xn = @(n) 1./(n+1);
    jacx = j_polynomial(length(xx),i,1,1,xx);
    jacy = j_polynomial(length(yy),j,1,1,yy);
    sol = xn(i).*(1-xx.^2).*jacx(:,end).* xn(j).*(1-yy.^2).*jacy(:,end);
    sol = reshape(sol,size(x));
end

function sol = phi_dx(i,j,x,y)
    xx = x(:); yy = y(:);
    xn = @(n) 1./(n+1);
    jacx = j_polynomial(length(xx),i,1,1,xx);
    if (i == 0)
        jacx_dx = 0*xx;
    else
        jacx_dx = 0.5*(i+3)*j_polynomial(length(xx),i-1,2,2,xx);
    end
    jacy = j_polynomial(length(yy),j,1,1,yy);
    sol = -xn(i).*(2*xx).*jacx(:,end).* xn(j).*(1-yy.^2).*jacy(:,end) + ...
        xn(i).*(1-xx.^2).*jacx_dx(:,end).* xn(j).*(1-yy.^2).*jacy(:,end);
    sol = reshape(sol,size(x));
end

function sol = phi_dy(i,j,x,y)
    xx = x(:); yy = y(:);
    xn = @(n) 1./(n+1);
    jacx = j_polynomial(length(xx),i,1,1,xx);
    jacy = j_polynomial(length(yy),j,1,1,yy);
    if (j == 0)
        jacy_dy = 0*yy;
    else
        jacy_dy = 0.5*(j+3)*j_polynomial(length(yy),j-1,2,2,yy);
    end
    sol = -xn(i).*(1-xx.^2).*jacx(:,end).* xn(j).*(2*yy).*jacy(:,end) + ...
        xn(i).*(1-xx.^2).*jacx(:,end).* xn(j).*(1-yy.^2).*jacy_dy(:,end);
    sol = reshape(sol,size(x));
end
    
function sol = evalsmooth(U,x,y)
    sol = 0;
    xx = x(:); yy = y(:);
    xn = @(n) 1./(n+1);
    jacx = j_polynomial(size(xx),size(U,1)-1,1,1,xx);
    jacy = j_polynomial(size(yy),size(U,2)-1,1,1,yy);
    for i = 1:size(U,1)
        for j = 1:size(U,2)
            phi_ij = xn(i-1).*(1-xx.^2).*jacx(:,i).* xn(j-1) ...
                .*(1-yy.^2).*jacy(:,j);
            sol = sol + U(i,j)*phi_ij;
        end
    end
    sol = reshape(sol,size(x));
end
