% script pour programmer la NMF
% r�alise les affichages
% les parties de cr�ation des donn�es et de mise � jour sont �
% compl�ter
% TP NMF / B. David & P. Magron, Janvier 2015

clear all; 
close all;

% cr�ation des donn�es 
Fe = 8000;
Nfft = 512;
 
% construction de la base spectrale W
% -----------------------------------
fnote = [100 232] ; % fr�quence des notes, do mi par exemple, en Hz
          % � compl�ter  
fnote = fnote/Fe; % fr�quences r�duites
R = length(fnote); % nombre de notes

% ici le calcul de W
Ws = zeros(Nfft/2,R);
for r=1:R
    % ici le calcul du spectre pour note r
    Wr = zeros(Nfft/2,1); % � compl�ter
    Ws(:,r)=Wr;
end

% calcul des activations
%-----------------------
Fh = Fe/(Nfft/4); % fr�q. d'�chantillonnage pour les activations
DeltaT =0.5; % en seconde
DeltaT = round(DeltaT*Fh);
t = (0:DeltaT-1)/Fh;
tau = 0.3 ; % � compl�ter
h = exp(-t/tau);
z = zeros(1,DeltaT);

Hs = [h z z;z h z] ; % � compl�ter � l'aide de h et z. 

% calcul de la repr�sentation de d�part
Xs = Ws*Hs;

% affichage
F = size(Ws,1);
T = size(Hs,2);
freq = (0:F-1)/Nfft*Fe;
ts = (0:T-1)/Fe;

figure
imagesc(ts,freq,Xs);axis xy
xlabel('temps')
ylabel('fr�quence')
title('Repr�sentation temps/fr�quence originale');

pause


% calcul de la NMF
% ----------------

% initialisation
W = zeros(size(Ws)); % � compl�ter
H = zeros(size(Hs)); % � compl�ter 

Nit = 100; % nombre d'it�ration

figure('position',[200 50 800 600]);
ha = annotation('textbox',[7 85 5 5]/100,'string','0');
annotation('textbox',[5 92 10 5]/100,'string','# Iterations','edgecolor','none');

for k=1:Nit
    
    V = W*H; % calcul de la repr�sentation approch�e
    
    % affichage
    subplot('position',[25 8 70 67]/100)
    imagesc(ts,freq,V);axis xy;set(gca,'yticklabel',[])
    xlabel('temps (s)')
    
    subplot('position',[10 8 12 67]/100)
    imagesc(1:size(H,1),freq,W);
    title('Atomes $\mathbf{W}$','interpreter','latex')
    axis xy;ylabel('fr�quence (Hz)')
    
    subplot('position',[25 80 70 15]/100)
    imagesc(ts,1:size(H,1),H);
    title('Activations temporelles $\mathbf{H}$','interpreter','latex')
    set(gca,'xticklabel',[]);set(gca,'ytick',1:size(H,1))
    set(ha,'string',num2str(k));
    drawnow
    pause(0.1)
    
    % update de H
    H = H ; % � modifier
    
    % update de W
    W = W ; % � modifier
    

end
