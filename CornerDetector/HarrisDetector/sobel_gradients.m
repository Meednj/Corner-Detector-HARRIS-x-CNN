function [Ix, Iy] = sobel_gradients(I)

    Sx = [-1 0 1; -2 0 2; -1 0 1];
    Sy = [-1 -2 -1; 0 0 0; 1 2 1];

    Ix = conv2(I, Sx, 'same');
    Iy = conv2(I, Sy, 'same');

end