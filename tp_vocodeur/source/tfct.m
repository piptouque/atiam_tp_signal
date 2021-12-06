% Analyse d'un signal � l'aide de la TFCT
% Transform�e de Fourier � Court Terme

clear all; close all;

% Param�tre d'interface   - Affichage du spectrogramme
affich = 1 ; % affich = 0 - Sans affichage
             % affich = 1 - Avec affichage
             % Note: cf. spectrogram sous matlab
             
% Param�tres du Vocodeur
M  = 1024;       % Longueur de la fen�tre
R  = M/4;        % Incr�ment des temps d'analyse, appel� hop size, t_a=pR
N  = 1024;       % Ordre de la tfd
w  = hanning(M); % D�finition de la fenetre d'analyse de taille M
ws = w;          % D�finition de la fen�tre de synth�se

if (N < M) 
    fprintf(2, 'N (%d) doit �tre plus grand que M (%d). \n\n', N, M); 
    return;
end

% Lecture du signal sonore et initialisation des param�tres associ�s
[x,Fe]  = audioread('salsa.wav');
x       = x(1:10*Fe, 1); x = x(:); % Signal audio, vecteur colonne monovoie
                                   % (voie gauche si st�r�o)
Nx      = length(x);               % Longueur du signal en �chantillons
Nt      = fix((Nx-M)/R);           % Nombre de trames

% Allocation m�moire
y = zeros(Nx, 1); % Signal de synth�se
if affich         % TFCT en cas d'affichage
    Xtilde = zeros(N, Nt);  
end

for p = 1:Nt              % Boucle sur les trames
   deb = (p-1)*R + 1;     % D�but de trame
   fin = deb + M - 1;     % Fin de trame
   tx  = x(deb:fin) .* w; % Extraction de la trame pond�r�e par la fen�tre
   X   = fft(tx, N);      % Tfd � l'instant b
   
   if affich, Xtilde(:, p) = X; end
   
   % D�but des op�rations de transformation de TFCT X pour obtenir Y
   % .... (par d�faut: pas de transformation, i.e. Y = X;
   
   Y = X;
   % Fin des op�rations de transformation de TFCT X pour obtenir Y
   
   % Calcul de ys � partir de Y (� compl�ter)
   % .....
%   y(deb:fin) = y(deb:fin) + ys(1:M);   % (OLA) Addition recouvrement
end

%soundsc(y, Fe) % � d�commenter pour jouer le son

% Affichage du spectrogramme
if affich    
    freq = Fe * (0:N/2)/N;            % �chelle des fr�quences en Hz
    b    = ((M-1)/2 + R*(0:Nt-1))/Fe; % Instants d'analyse (au centre de la
                                      % fen�tre)
    imagesc(b, freq, db(abs(Xtilde(1:(N/2+1), :))));
    axis xy;
end