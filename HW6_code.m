Y=table2array(card(:,6));
x=table2array(card(:,1:5));
X=[ones(size(Y)),x];
[beta,Sigma,resid,CovB]=mvregress(X,Y,'varformat','full');
beta_MH=repmat([beta',Sigma],[1 1 2]);
acc=zeros(2,1);
vcov=transpose(diag(CovB));
vcov1=vcov*8e-03;
vcov2=vcov*5e-03;
for j=2:5000
    lb=L1(Y,X,beta_MH(j-1,1:end-1,1),beta_MH(j-1,end,1));
    betaNew=beta_MH(1,:,1)+mvnrnd(zeros(size(beta_MH(1,:,1))),vcov1);
    lbNew=L1(Y,X,betaNew(1:end-1),betaNew(end));
    d=lbNew-lb;
    c=log(rand);
    if(d>=c)
        beta_MH(j,:,1)=betaNew;
        acc(1,1)=acc(1,1)+1;
    else
        beta_MH(j,:,1)=beta_MH(j-1,:,1);
    end
end
for i=2:5000
    lb=L2(Y,X,beta_MH(i-1,1:end-1,2),beta_MH(i-1,end,2));
    betaNew=beta_MH(i-1,:,2)+mvnrnd(zeros(size(beta_MH(1,:,2))),vcov2);
    lbNew=L2(Y,X,betaNew(1:end-1),betaNew(end));
    d=lbNew-lb;
    c=log(rand);
    if(d>=c)
        beta_MH(i,:,2)=betaNew;
        acc(2,1)=acc(2,1)+1;
    else
        beta_MH(i,:,2)=beta_MH(i-1,:,2);
    end
end
titl={'All priors flat','All priors flat except educ'};
str={'Const','educ','black','smsa','south','exper','Sigma'};
for i=1:2
    figure('Name',titl{i});
    for j=1:7
        subplot(4,2,j);
        histogram(beta_MH(:,j,i),50);
        title(str{j});
    end
end
 function l=L1(y,x,beta,sigma)
u=y-x*beta';
sig=sqrt(sigma);
R=normpdf(u,0,sig);
l=-sum(log(R));
 end   
function l=L2(y,x,beta,sigma)
u=y-x*beta';
sig=sqrt(sigma);
R=normpdf(u,0,sig);
l=-sum(log(R))-log(normpdf(beta(2),0.06,0.025));
end