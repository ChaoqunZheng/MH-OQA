function idx = icm(Cx, X, Cy, Y, mu1,mu2)

if nargin == 2
    dist = sqdist(Cx, X);
    [~, idx] = min(dist); 
elseif nargin == 6
    dist = mu1 * sqdist(Cx, X) + mu2 * sqdist(Cy, Y);
    [~, idx] = min(dist);
else
    error('Wrong number of input arguments.');
end

end
