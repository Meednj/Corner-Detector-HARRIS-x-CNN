function coins = non_max_suppression(R, seuil, taille)

    coins = zeros(size(R));

    for i = 1+floor(taille/2) : size(R,1)-floor(taille/2)
        for j = 1+floor(taille/2) : size(R,2)-floor(taille/2)

            window = R(i-floor(taille/2):i+floor(taille/2), ...
                       j-floor(taille/2):j+floor(taille/2));

            if R(i,j) == max(window(:)) && R(i,j) > seuil
                coins(i,j) = 1;
            end
        end
    end

end