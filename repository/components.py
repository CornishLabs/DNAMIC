from ndscan.experiment import (
    ExpFragment,
    FloatParam, IntParam,
    IntChannel,
    MHz, us, ms
)
import random, numpy as np
import time
from repository.models.atom_response import image_from_probs_and_locs

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
        self.setattr_param("threshold",   IntParam,   "Threshold",   default=600,    min=0)

        self.setattr_result("counts",          IntChannel)
        self.setattr_result("is_bright_class", IntChannel)

    def run_once(self):
        pb  = self.p_bright.get()
        thr = self.threshold.get()

        image = image_from_probs_and_locs([(16,16,pb),(24,16,pb),(32,16,pb)])

        try:
            rois = self.get_dataset("rois",archive=False)
        except KeyError:
            # sensible default: three 4×4 boxes centered on your sites
            rois = [(14,18,14,18), (14,18,22,26), (14,18,30,34)]
            self.set_dataset("rois", rois, broadcast=True)

        # Sum counts in each ROI and classify
        y0,y1,x0,x1 = rois[0]
        cts = int(image[y0:y1, x0:x1].sum()) 
        classified = int(cts >= thr) # [int(c >= thr) for c in cts]

        self.set_dataset("last_image",image,broadcast=True)

        print(f"[Readout] p_b={pb:.3f} | cts={cts} thr={thr} → {'B' if classified else 'D'}")

        self.counts.push(cts)
        self.is_bright_class.push(classified)