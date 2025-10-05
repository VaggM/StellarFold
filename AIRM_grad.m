function grad = AIRM_grad(X1, X2)

    isqX1 = inv(X1);
    [u, e] = schur(isqX1*X2*isqX1);
    e = diag(e);
    grad = 2*isqX1*u*diag(log(e)./e)*u'*isqX1;

end
