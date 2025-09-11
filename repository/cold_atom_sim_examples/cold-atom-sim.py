from ndscan.experiment import (
    ExpFragment, kernel, rpc,
    FloatParam, IntParam,
    IntChannel, FloatChannel, OpaqueChannel,
    MHz, us, ms,
    make_fragment_scan_exp, SubscanExpFragment,
    LinearGenerator, ScanOptions, CustomAnalysis
)
import math, random, numpy as np

from models.atom_response import p_bright_detuned_rabi
from components import PrepareAtom, Pulse, ReadoutFluorescence

class OneShot(ExpFragment):
    def build_fragment(self):
        self.setattr_fragment("prep",  PrepareAtom)
        self.setattr_fragment("pulse", Pulse)
        self.setattr_fragment("ro",    ReadoutFluorescence)

        # Lineshape parameters
        self.setattr_param("resonance_position", FloatParam, "Resonance location", default=10.0*MHz, unit="MHz")
        self.setattr_param("rabi_freq",     FloatParam, "Rabi frequency",  default=1*MHz, unit="MHz", min=0.0)
        self.setattr_param("shot_index", IntParam, "Shot index", default=0, is_scannable=True)

        # Efficient handle to set p_bright
        _, self._pb_store = self.ro.override_param("p_bright")

    def run_once(self):
        self.prep.run_once()
        self.pulse.run_once()

        # Simulate atom response (state -> p_bright)
        pb = p_bright_detuned_rabi(
            self.pulse.frequency.get(),        # Hz
            self.resonance_position.get(),     # Hz
            self.rabi_freq.get(),              # Hz
            self.pulse.duration.get(),         # seconds
        )
        self._pb_store.set_value(pb)

        self.ro.run_once()

    def get_default_analyses(self):
        return [
            CustomAnalysis([self.shot_index], self._analyse_shots_to_p, [
                FloatChannel("p", "Bright probability"),
                FloatChannel("p_err", "Bright probability Average Error", display_hints={"error_bar_for": "_p","priority":2}),
                FloatChannel("p_upper_err", "Bright probability Upper Error", display_hints={"priority": -10}),
                FloatChannel("p_lower_err", "Bright probability Lower Error", display_hints={"priority": -11}),
            ])
        ]

    def _analyse_shots_to_p(self, axis_values, result_values, analysis_results):
        classes = result_values[self.ro.is_bright_class]
        n = len(classes)
        k = int(sum(1 for v in classes if v))
        p_hat = (k / n) if n > 0 else 0.0

        # Wilson interval (asymmetric), with z from param
        z = 1
        if n > 0:
            denom = 1.0 + (z*z)/n
            center = (p_hat + (z*z)/(2*n)) / denom
            radius = (z * math.sqrt(p_hat*(1.0 - p_hat)/n + (z*z)/(4*n*n))) / denom
            lower = max(0.0, center - radius)
            upper = min(1.0, center + radius)
        else:
            lower, upper = 0.0, 1.0

        p_upper_err = upper - p_hat
        p_lower_err = p_hat - lower

        analysis_results["p"].push(p_hat)
        analysis_results["p_err"].push((p_upper_err+p_lower_err)/2)
        analysis_results["p_upper_err"].push(p_upper_err)
        analysis_results["p_lower_err"].push(p_lower_err)

        return []


SingleShotExperiment = make_fragment_scan_exp(OneShot)

class ShotScan(SubscanExpFragment):
    pass

class ShotChunk(ExpFragment):
    def build_fragment(self) -> None:
        self.setattr_fragment("one_shot", OneShot)
        self.setattr_fragment("shot_scan", ShotScan, self, "one_shot", [(self.one_shot, "shot_index")])
        
        self.setattr_param("shots_per_chunk", IntParam, "Shots per chunk", default=10, min=1)

    def configure_scan(self):
        if self.shots_per_chunk.changed_after_use():    
            N = self.shots_per_chunk.get()
            gen = LinearGenerator(0, N-1, N, True)
            opts = ScanOptions(
                num_repeats=1,
                num_repeats_per_point=1,
                randomise_order_globally=False
            )
            self.shot_scan.configure([(self.one_shot.shot_index, gen)], options=opts)

    def host_setup(self):
        self.configure_scan()
        super().host_setup()

    def device_setup(self):
        # Update scan if shots_per_chunk was changed (can be left out if
        # there are no scannable parameters influencing the scan settings).
        self.configure_scan()
        self.device_setup_subfragments()

    def run_once(self):
        self.shot_scan.run_once()

MultiShotExperiment = make_fragment_scan_exp(ShotChunk)