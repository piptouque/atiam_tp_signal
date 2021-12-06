import("stdfaust.lib");

//---------------------------------------------------------
// volterra2021.dsp
// Thomas Helie, Yann Orlarey
//---------------------------------------------------------

// Main process
process = source : drive(fmoog) : output <: _,_;

// Moog filter 
fmoog = _,0 : M : M : M : M : S
    with {
        // linear filter
        F1 = *(v/(1+v)) : + ~ *(1/(1+v)) with {
            v = 2*ma.PI*hslider("h:moog/freq", 4400, 20, 10000, 1)/ma.SR;
        };
        // module stage
        M   = (_ <: F1, ^(3)), _
                : (_ <: _, (^(3) : *(-1))), +
                : _, (+ : F1);
        // summation stage with optional non-linearity
        S = _, *(checkbox("h:moog/NL") * -1/3) : +;
    };

// Drive system 
drive(fx) = *(g) : fx : /(g) 
with { 
    g = hslider("v:drive/drive", 2, 1, 5, 0.01); 
};

// Test source made of two slighlty detuned square signals
source = os.square(f-d/2)+os.square(f+d/2) : /(2)
with { 
    f = hslider("v:[1]source/squarefreq", 220, 20, 2000, 1);
    d = hslider("v:[1]source/squaredelta", 2, 0.05, 10, 0.01);
};

// output stage with a large attenuation control
output = /(att)
with {
    att = hslider("v:output/att[scale:log]", 5, 1, 10000, 0.1); 
};