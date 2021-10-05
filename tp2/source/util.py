

import numpy as np
import scipy
import scipy.io
from scipy.signal import lfilter
import soundfile as sf
import os
import pyaudio
import wave


def play(y, Fe=44100):
    z = (np.real(y)/(abs(np.real(y)).max())).astype(np.float16)

    # see: http://people.csail.mit.edu/hubert/pyaudio/docs/
    # and: https://stackoverflow.com/a/27961508
    number_channels = z.shape[1] if len(z.shape) == 2 else 1
    print(Fe)
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(z.itemsize),
        channels=number_channels,
        rate=Fe,
        output=True)

    stream.write(z)

    stream.stop_stream()
    stream.close()

    p.terminate()


def moindres_carres(p, x, z):
    """ 
    % Cette fonction renvoie simplement le filtre h qui 
    % minimise l'énergie de 
    % z-h*x    (h*x est la convolution PAS LE PRODUIT de h et x)
    % h est de taille p+1
    % Si la taille de z n'est pas suffisante, le programme se permet de tronquer
    % les signaux en conséquence.
    % x et z sont traités comme des signaux causaux
    """
    assert (len(x.shape) == 2 and min(x.shape) == 1) or len(
        x.shape) == 1, "La taille de x est mauvaise"
    assert (len(z.shape) == 2 and min(z.shape) == 1) or len(
        z.shape) == 1, "La taille de z est mauvaise"
    xc = x.copy().reshape(-1)  # colonne
    zc = z.copy().reshape(-1)
    # """ xmat est une matrice telle que h@xmat est h*x (h convolué avec x)"""
    xmat = np.zeros((p+1, len(xc)+p))
    for k in range(p+1):
        xmat[k, k:k+len(x)] = xc

    # si z est trop long
    if len(zc) > len(xc)+p:
        zc = zc[:len(xc)+p]
    # si z est trop court
    if len(zc) < len(xc)+p:
        zc = np.concatenate((zc, np.zeros(len(xc)+p-len(zc))))
    # resolution du problème : trouver h telle que norm(z-h@xmat) soit minimale
    # Vous pouvez essayer de démontrer la formule hors du TP
    L = xmat@(xmat.T)
    V = xmat@zc.T
    h = np.linalg.inv(L)@V
    return h


def lpc_morceau(p, x):
    """ % Renvoie un filtre de taille p+1 dont la première valeur est 1 et qui
% minimise le h*x
"""
    assert (len(x.shape) == 2 and min(x.shape) == 1) or len(
        x.shape) == 1, "La taille de x est mauvaise"
    xc = x.copy().reshape(-1)  # colonne
    # """ xmat est une matrice telle que h@xmat est h*x (h convolué avec x)"""
    xmat = np.zeros((p+1, len(xc)+p))
    for k in range(p+1):
        xmat[k, k:k+len(x)] = xc
    # n=np.arange(0,len(xc))
    # xc=xc*cos(pi*n/len(xc))

    # resolution du problème : trouver h telle que norm(z-h@xmat) soit minimale
    # Vous pouvez essayer de démontrer la formule hors du TP
    M = xmat@(xmat.T)
    L = M[1:, 1:]
    V = M[1:p+1, 0]
    h1 = -np.linalg.inv(L)@V
    h = np.concatenate((np.ones(1), h1))
    return h


def norm(X):
    return ((X**2).sum())**0.5
