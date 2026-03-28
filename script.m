I = imread('image.png');
%size(I);
subplot(1,2,1)
imshow(I)
title('Image Originale')
if size(I,3) == 3
    I = rgb2gray(I);
end
I = double(I);
subplot(1,2,2)
imshow(I)
title('Image en niveaux de gris')


Sx = [-1 0 1;
      -2 0 2;
      -1 0 1];
Sy = [-1 -2 -1;
       0  0  0;
       1  2  1];
Ix = conv2(I, Sx, 'same');
Iy = conv2(I, Sy, 'same');

figure
subplot(1,2,1)
imshow(Ix,[])
title('Gradient horizontal Ix')
subplot(1,2,2)
imshow(Iy,[])
title('Gradient vertical Iy')

%les produit des gradient pour la matrice de convolution
Ix2 = Ix.^2;
Iy2 = Iy.^2;
Ixy = Ix .* Iy;

sigma = 2;
sizeG = 7;
[x,y] = meshgrid(-floor(sizeG/2):floor(sizeG/2), -floor(sizeG/2):floor(sizeG/2));
g = exp(-(x.^2 + y.^2)/(2*sigma^2)); %fct guassienne
g = g / sum(g(:)); %normalisation du filtre guassian

Sx2 = conv2(Ix2, g, 'same');
Sy2 = conv2(Iy2, g, 'same');
Sxy = conv2(Ixy, g, 'same');
figure
subplot(1,3,1)
imshow(Sx2,[])
title('Sx2')

subplot(1,3,2)
imshow(Sy2,[])
title('Sy2')

subplot(1,3,3)
imshow(Sxy,[])
title('Sxy')

%reponse de harris
k = 0.04;
R = (Sx2 .* Sy2 - Sxy.^2) - k * (Sx2 + Sy2).^2;
figure
imshow(R,[])
title('Réponse de Harris')
colorbar