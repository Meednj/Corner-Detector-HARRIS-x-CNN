function g = gaussian_kernel(sizeG, sigma)

    [x,y] = meshgrid(-floor(sizeG/2):floor(sizeG/2), ...
                     -floor(sizeG/2):floor(sizeG/2));

    g = exp(-(x.^2 + y.^2)/(2*sigma^2));
    g = g / sum(g(:));

end