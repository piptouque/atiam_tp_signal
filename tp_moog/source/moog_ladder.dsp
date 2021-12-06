import("stdfaust.lib");

// process = timbre(n, f_0) * vol * (gate : fi.lowpass(2, 10)) : echo(d, fb);


H3 = _, 0 : seq(i, 4, F3) : (_:T3), _ : + with {
    F1 = *(nu / (1 + nu)) : + ~ (* (1 / (1 + nu))) with {
        f_c = hslider("h:filter/cutoffreq",  440, 10, 2000, 0.1);
        nu = 2 * ma.PI * f_c / ma.SR;
    };
    F3 = (_, (_ <: (^(3), (F1 <: ^(3), _)))) : ((_, (_, _ : -) : + : F1), _) : ro.cross(2);
    T3 = *(-1/3);
};


pan(s, a) = s*(2 * a), s*(2 * (1 - a));
inst = pan(timbre(n, f_0) * gain * gate, panning)
    with {
        // n = hslider("h:inst/num_harmonics", 2, 0, 6, 1);
        n = 4;
        osc = _, ma.SR : / : + ~ _ : _, 2*ma.PI  : * : sin;
        vol_harm(i) = vslider("h:inst/harm %i", 0.1, 0, 1, 0.01);
        timbre(n, f) = sum(i, n, osc(f*(i+1))*vol_harm(i));
        f_0 = hslider("h:inst/freq", 440, 20, 8000, 0.1);
        gain = vslider("h:inst/gain", 0.2, 0, 1, 0.1);
        panning = hslider("h:inst/pan", 0.5, 0, 1, 0.1);
        gate = button("h:inst/gate");     
};
global_gain = vslider("h:filter/global_gain", 0.2, 0, 1, 0.1);

source = inst;
// source = par(i, 2, H3);
// source = inst : _, H3;



process = source : par(i, 2, H3 : *(global_gain));
