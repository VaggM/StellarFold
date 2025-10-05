function dist = AIRM(X, B)

    e = eig(X, B);
    lge = log(e);
    dist = real(sum(lge.^2));

end
