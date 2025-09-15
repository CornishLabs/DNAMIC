from ndscan.experiment import (
    ExpFragment, kernel, rpc,
    FloatParam, IntParam,
    IntChannel, FloatChannel, OpaqueChannel,
    MHz, us, ms,
    make_fragment_scan_exp, CustomAnalysis, annotations,
    SubscanExpFragment, ScanOptions, LinearGenerator
)

from repository.sequences.atom_MW_state_change import MultiShotAnalysed


class ScanFrequency(SubscanExpFragment):
    def build_fragment(self):
        self.setattr_fragment("one_p", MultiShotAnalysed)
        super().build_fragment(self, "one_p", [(self.one_p.carrier.shot.pulse, "frequency")])


    def _configure(self):
        gen  = LinearGenerator(5*MHz, 15*MHz, 20, True)
        opts = ScanOptions(num_repeats=1, num_repeats_per_point=1, randomise_order_globally=True)
        self.configure([(self.one_p.carrier.shot.pulse.frequency, gen)], options=opts) 

    def host_setup(self):
        self._configure()
        super().host_setup()

    def device_setup(self):
        self._configure()
        self.device_setup_subfragments()


ScanFrequencyExperiment = make_fragment_scan_exp(ScanFrequency)