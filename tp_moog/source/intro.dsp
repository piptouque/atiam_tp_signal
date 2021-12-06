import("stdfaust.lib");
// process = (1, 2): +;
// process = 1, ((2, 3): +): +;
// process = (3, (1, 2:+)):-;
a = 1;
b = 4;
c = -;
// process = a, b: c;

// Exemple d'utilisation de composition récursive :
// process = 1: +~_;
// Ça créé un signal z(n) tq z(n) = z(n-1) + 1
// Ici l'opérateur ~ a le même rôle que : et , !!
//
// Déf de la partie décimale :
dec = _, 1: fmod;
// signal en dent-de-scie :
saw_a_offset = 0.01:(+:dec)~_;
// normalisation :
normalise_two = _,0.5:-,2:*;
saw_a = saw_a_offset:normalise_two;
// process = saw_a;
// avec une fréq en entrée
f_s = ma.SR;
// syntaxe sans variable / fonction
saw_offset = (_, f_s: /):(+:dec)~_;
// syntaxe fonction
saw(f) = f:saw_offset:normalise_two;
normalise(s, amp, offset) = s,offset:-,amp:/; 
f_0 = hslider("freq", 440, 20, 8000, 0.1);
vol = vslider("gain", 0.2, 0, 1, 0.05);
panning = hslider("pan", 0.5, 0, 1, 0.01);
gate = button("gate");
f_1 = f_0 * 3/2;
f_2 = f_0 * 6/5;
f_3 = f_0 * 8/5;
// process = normalise(saw(f_0)+saw(f_1)+saw(f_2)+saw(f_3), 4, 0) * vol;


delay = 10000;

echo(d, fb) = + ~ (@(d) : *(fb));

osc(f) = (f/f_s) : +~_ : *(ma.PI*2) : sin;
pan(s, val) = s*(2 * val), s*(2 * (1 - val));
// process = pan(osc(f_0) * vol * gate, panning);

d = 10000;
fb = 0.2;

vol_harm(i) = vslider("h:toto/harm %i", 0.1, 0, 1, 0.01);

timbre(n, f) = sum(i, n, osc(f*(i+1))*vol_harm(i));
n = 5;
process = timbre(n, f_0) * vol * (gate : fi.lowpass(2, 10)) : echo(d, fb);