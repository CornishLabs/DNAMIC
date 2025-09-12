from ndscan.experiment import (
    ExpFragment, kernel, rpc,
    FloatParam, IntParam,
    IntChannel, FloatChannel, OpaqueChannel,
    MHz, us, ms,
    make_fragment_scan_exp
)

from models.atom_response import p_bright_detuned_rabi
from components import PrepareAtom, Pulse, ReadoutFluorescence

from reusable.make_shot_scan import make_shot_chunk_exp_fragments_from_shot
from reusable.single_shot_base import SingleShotBase

class OneShot(SingleShotBase):
    def build_fragment(self):
        self.setattr_fragment("prep",  PrepareAtom)
        self.setattr_fragment("pulse", Pulse)
        self.setattr_fragment("ro",    ReadoutFluorescence)

        # Lineshape parameters
        self.setattr_param("resonance_position", FloatParam, "Resonance location", default=10.0*MHz, unit="MHz")
        self.setattr_param("rabi_freq",     FloatParam, "Rabi frequency",  default=1*MHz, unit="MHz", min=0.0)

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

    def get_classification_handle(self):
        return self.ro.is_bright_class

    def get_counts_handle(self):
        return self.ro.counts


OneShotCarrier, MultiShot = make_shot_chunk_exp_fragments_from_shot(OneShot)
MultiShotExperiment = make_fragment_scan_exp(MultiShot)
