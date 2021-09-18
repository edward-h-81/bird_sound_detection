(
{ SinOsc.ar(
    SinOsc.ar(
        XLine.kr(1, 100, 5)).exprange(*XLine.kr([20, 800], [7000, 200], 10)
    )
   ) * 0.5

}.play;
s.record(duration: 10);
)



(
SynthDef("help-PinkNoise", { arg out=0;
    Out.ar(out,
        PinkNoise.ar(0.6)
    )
}).play;
s.record(duration: 10);
)