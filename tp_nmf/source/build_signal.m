%  Signal synthesis from piano isolated notes
%  B. David, P. Magron, january 2015.
% 
% Inputs :
%     notes : matrix of notes with 4 lines : pitch, t onset, durations and
%     velocity
%     Fs : sample rate (in Hz)
%
% Outputs :
%     x : synthesis signal


function x = build_signal(notes,Fs)

pitch = notes(1,:);
ton = notes(2,:);
deltaT = notes(3,:);
vel = notes(4,:);

Nnotes = length(pitch);
Lsig = floor((ton(end)+deltaT(end))*Fs);
x = zeros(1,Lsig);

for n=1:Nnotes
    
    % Get midi index
    ind_pitch = pitch(n);
    Lnote = floor(deltaT(n)*Fs);
        
    % Read piano note
    [xt,Fs_old] = audioread(strcat('data/',int2str(ind_pitch),'.wav'));
    xt = resample(xt,Fs,Fs_old)';
    xt = xt(1:Lnote) * vel(n);

    % Fill the current melodic line
    deb = ton(n)*Fs+1  ;  fin = deb + Lnote-1;
    x(deb:fin) = x(deb:fin) + xt;
end

end

