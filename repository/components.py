from ndscan.experiment import (
    ExpFragment,
    FloatParam, IntParam,
    IntChannel,
    MHz, us, ms
)
import random, numpy as np
import time

class PrepareAtom(ExpFragment):
    def build_fragment(self):
        self.setattr_param("cool_time", FloatParam, "Cooling time", default=2.0*ms, min=0.0, unit="ms")
    
    def run_once(self):
        ct = self.cool_time.get()
        time.sleep(ct)
        print(f"[Prepare] {self.cool_time.get()/ms:.1f} ms → prepared")

class Pulse(ExpFragment):
    def build_fragment(self):
        self.setattr_param("frequency",   FloatParam, "Drive freq",    default=10.0*MHz, unit="MHz")
        self.setattr_param("duration", FloatParam, "Duration", default=0.48*us,  unit="us", min=0.0)

    def run_once(self):
        print(f"[Pulse] f={self.frequency.get()/MHz:.3f} MHz, t={self.duration.get()/us:.2f} µs")

class ReadoutFluorescence(ExpFragment):
    def build_fragment(self):
        self.setattr_param("p_bright",    FloatParam, "Bright prob", default=0.5, min=0.0, max=1.0)
        self.setattr_param("mean_bright", FloatParam, "Mean bright counts", default=1200.0, min=0.0)
        self.setattr_param("mean_dark",   FloatParam, "Mean dark counts",  default=150.0,  min=0.0)
        self.setattr_param("threshold",   IntParam,   "Threshold",   default=600,    min=0)

        self.setattr_result("counts",          IntChannel)
        self.setattr_result("is_bright_class", IntChannel)

    def run_once(self):
        pb  = self.p_bright.get()
        muB = self.mean_bright.get()
        muD = self.mean_dark.get()
        thr = self.threshold.get()

        is_bright  = (random.random() < pb)
        lam        = muB if is_bright else muD
        cts        = int(np.random.poisson(lam=max(0.0, lam)))
        classified = int(cts >= thr)

        print(f"[Readout] p_b={pb:.3f} → {'B' if is_bright else 'D'} | cts={cts} thr={thr} → {'B' if classified else 'D'}")

        self.counts.push(cts)
        self.is_bright_class.push(classified)