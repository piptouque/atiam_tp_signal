import("stdfaust.lib");


declare moog_name "Moog Ladder";
declare moog_author "Hartenstein Matthieu, Thiel Pierre";
declare moog_version "1.0";
declare moog_license "MIT License (MIT)";
declare description "Moog Ladder Assigment using Volterra series";



H3 = _, 0 : seq(i, 4, M) : _, (_:T3:*(moog_nl)) : + with {
    // Linéarisation de (F)
    F1 = *(nu / (1 + nu)) : + ~ (* (1 / (1 + nu))) with {
        nu = 2 * ma.PI * moog_f_c / ma.SR;
    };
    // Version simplifiée d'un étage de (F) à l'ordre 3
    M = ((_ <: ((F1 <: _, ^(3)), (^(3)))),_) : (_, ((_, _ : ro.cross(2) : -), _ : + : F1));
    // Coefficient du DL3 de tanh
    T3 = *(-1/3);

    moog_f_c = hslider("h:system/cutoffreq", 2000, 10, 20000, 0.1);
    moog_nl = checkbox("h:system/NL");
};


// Applique un gain avant et le retire après un système donné.
// On remarque qu'un système linéaire reste inchangé.
// Cela permet d'observer les non-linéarité
// de la sortie du système sans a priori augmenter
// son volume.
moog_drive = hslider("h:system/drive", 0.5, 0.1, 3, 0.01);
drive(fx) = *(moog_drive) : fx : /(moog_drive);


// Un instrument simple généré par synthèse additive.
inst = timbre(n, f_0) : *(gate)
    with {
        // n = hslider("h:inst/num_harmonics", 2, 0, 6, 1);
        n = 4;
        osc = _, ma.SR : / : + ~ _ : _, 2*ma.PI  : * : sin;
        vol_harm(i) = vslider("h:inst/harm %i", 1, 0, 1, 0.01);
        timbre(n, f) = sum(i, n, osc(f*(i+1))*vol_harm(i));
        f_0 = hslider("h:inst/freq", 440, 20, 8000, 0.1);
        gain = vslider("h:inst/gain", 0, 0, 1, 0.1);
        
        // On pourrait ajouter un effet de spatialisation à l'instrument.
        // panning(s, a) = s*(2 * a), s*(2 * (1 - a));
        // pan = hslider("h:inst/pan", 0.5, 0, 1, 0.1);

        gate = button("h:inst/gate");     
};

//
// Test source made of two slighlty detuned square signals
square = os.square(f-d/2)+os.square(f+d/2) : /(2)
with { 
    f = hslider("v:square/freq", 220, 20, 2000, 1);
    d = hslider("v:square/delta", 2, 0.05, 10, 0.01);
};
global_gain = hslider("h:global/global_gain", 0.8, 0, 1, 0.1);

source = inst;
// source = square;

process = source : drive(H3) : *(global_gain) <: _,_;