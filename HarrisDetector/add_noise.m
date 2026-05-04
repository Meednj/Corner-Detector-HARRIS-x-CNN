function I_noise = add_noise(I, variance)

    noise = sqrt(variance) * randn(size(I));
    I_noise = I + noise;

end