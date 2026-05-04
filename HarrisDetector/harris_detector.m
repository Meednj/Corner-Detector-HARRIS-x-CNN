function [R, coins] = harris_detector(I, k, sigma, sizeG, seuil, taille)

    [Ix, Iy] = sobel_gradients(I);

    Ix2 = Ix.^2;
    Iy2 = Iy.^2;
    Ixy = Ix .* Iy;

    g = gaussian_kernel(sizeG, sigma);

    Sx2 = conv2(Ix2, g, 'same');
    Sy2 = conv2(Iy2, g, 'same');
    Sxy = conv2(Ixy, g, 'same');

    R = (Sx2 .* Sy2 - Sxy.^2) - k * (Sx2 + Sy2).^2;

    coins = non_max_suppression(R, seuil, taille);

end