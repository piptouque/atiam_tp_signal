function output = ola(w, hop, Nb)

% function output = ola(w,hop,Nb)
% R�alise l'addition-recouvrement d'une fen�tre w,
% avec un d�calage hop et un nombre de fenetres Nb de valeur 10 par d�faut.

if nargin <= 2, Nb = 10; end

w = w(:).';     % Vecteur ligne
N = length(w);
output = zeros(1, N + (Nb-1)*hop); % R�serve l'espace m�moire

for k = 1:Nb
    deb = (k-1)*hop + 1;
    fin = deb + N - 1;
    output(deb:fin) = output(deb:fin) + w; % OLA
end

