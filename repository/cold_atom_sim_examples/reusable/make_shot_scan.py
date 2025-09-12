from ndscan.experiment import (
    ExpFragment, SubscanExpFragment,
    IntParam, LinearGenerator, ScanOptions,
    CustomAnalysis, FloatChannel, make_fragment_scan_exp
)
import math

def make_shot_indexed_carrier(ShotCls):
    """Return a concrete carrier class wrapping `ShotCls` and owning `shot_index` + analysis."""
    class ShotCarrier(ExpFragment):
        def build_fragment(self):
            # axis lives HERE (so analysis can be here too)
            self.setattr_param("shot_index", IntParam, "Shot index", default=0, is_scannable=True)
            self.setattr_fragment("shot", ShotCls)

        def run_once(self):
            self.shot.run_once()

        # Summarise the N shots → p ± err (uses child’s declared handles)
        def get_default_analyses(self):
            return [
                CustomAnalysis(
                    [self.shot_index],
                    self._analyse_shots_to_p,
                    [
                        FloatChannel("p", "Bright probability"),
                        FloatChannel("p_err", "± error", display_hints={"error_bar_for": "_p", "priority": 2}),
                    ],
                )
            ]

        def _analyse_shots_to_p(self, axis_values, result_values, analysis_results):
            classes_handle = self.shot.get_classification_handle()
            classes = result_values[classes_handle]
            n = len(classes)
            k = int(sum(1 for v in classes if v))
            p_hat = (k/n) if n else 0.0

            z = 1.0  # ~68% CI
            if n:
                denom  = 1.0 + (z*z)/n
                center = (p_hat + (z*z)/(2*n)) / denom
                radius = (z * math.sqrt(p_hat*(1.0 - p_hat)/n + (z*z)/(4*n*n))) / denom
                lower, upper = max(0.0, center - radius), min(1.0, center + radius)
                p_err = max(upper - p_hat, p_hat - lower)
            else:
                p_err = 1.0

            analysis_results["p"].push(p_hat)
            analysis_results["p_err"].push(p_err)
            return []

    return ShotCarrier


def make_shot_chunk_exp_fragments_from_shot(ShotCls, *, default_shots_per_chunk=40):
    Carrier = make_shot_indexed_carrier(ShotCls)

    # class ShotScan(SubscanExpFragment):
    #     pass

    class ShotChunk(SubscanExpFragment):
        def build_fragment(self):
            self.setattr_fragment("carrier", Carrier)
            
            # IF I have the empty class above doing the scan then
            # self.setattr_fragment("shot_scan", ShotScan, self, "carrier", [(self.carrier, "shot_index")])
            # ELSE (Then I have this class owning the scanning)
            super().build_fragment(self, "carrier", [(self.carrier, "shot_index")])

            self.setattr_param("shots_per_chunk", IntParam, "Shots per chunk",
                               default=default_shots_per_chunk, min=1)

        def _configure(self):
            # NOTE: Perhaps there's a performance opt. with checking `self.shots_per_chunk.changed_after_use()`?
            N = self.shots_per_chunk.get()
            gen  = LinearGenerator(0, N-1, N, True)
            opts = ScanOptions(num_repeats=1, num_repeats_per_point=1, randomise_order_globally=False)
            self.configure([(self.carrier.shot_index, gen)], options=opts)  # self.shot_scan.configure if using empty class

        def host_setup(self):
            self._configure()
            super().host_setup()

        def device_setup(self):
            self._configure()
            self.device_setup_subfragments()

        # IF using empty class
        # def run_once(self):
        #     self.shot_scan.run_once()

    return Carrier, ShotChunk