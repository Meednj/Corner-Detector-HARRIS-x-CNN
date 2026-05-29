clear; close all;

I = imread('images/cube.png');

if size(I,3)==3
    I = rgb2gray(I);
end

I = double(I);

% parametres
k           = 0.05;
sigma       = 1.5;
sizeG       = 7;
taille      = 5;
seuil_ratio = 0.0002;

% image normale 
[R, coins] = harris_detector(I, k, sigma, sizeG, seuil_ratio, taille);

figure
imshow(I,[]); hold on
[y,x] = find(coins);
plot(x,y,'r+')
title('Coins - image normale')

% image bruitee
I_noise = add_noise(I, 0.02);
[Rn, coins_n] = harris_detector(I_noise, k, sigma, sizeG, seuil_ratio, taille);

figure
imshow(I_noise,[]); hold on
[y,x] = find(coins_n);
plot(x,y,'g+')
title('Coins - image bruitee')

% image eclaircie
I_bright = change_brightness(I, 1, 50);
[Rb, coins_b] = harris_detector(I_bright, k, sigma, sizeG, seuil_ratio, taille);

figure
imshow(I_bright,[]); hold on
[y,x] = find(coins_b);
plot(x,y,'b+')
title('Coins - image eclaircie')