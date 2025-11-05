from artiq.experiment import *
from artiq.coredevice.ad9910 import (
    RAM_DEST_ASF,
    RAM_MODE_DIRECTSWITCH,
    RAM_MODE_RAMPUP,
)

class UrukulToneRAM(EnvExperiment):
    """
    One RAM payload, three profiles:
      - P0: OFF       (DIRECTSWITCH to single 0.0 sample)
      - P1: SQUARE    (DIRECTSWITCH to single 1.0 sample)
      - P2: BH (8us)  (RAMPUP across BH segment; one-shot)
    Playback order (10x):
      SQUARE (2us) → OFF (1us) → BH (8us) → OFF (2us)
    Timing is controlled by ARTIQ profile-pin changes and delays.
    """

    # ==== EDIT THESE TO MATCH YOUR SETUP ====
    SYSCLK_HZ = 1_000_000_000     # AD9910 system clock (Hz). Commonly 1e9
    BH_DWELL_S = 100e-9           # Per-sample dwell inside the BH envelope
    CARRIER = 1*MHz               # RF carrier (since we amplitude modulate)
    ATT = 10.0*dB                 # Output attenuation
    # ========================================

    def build(self):
        self.setattr_device("core")
        self.dds = self.get_device("urukul4_ch0")
        self.cpld = self.get_device("urukul4_cpld")

        # host-side buffers
        self.ram_amp = []         # float amplitudes 0.0..1.0
        self.asf_words = []       # packed ints
        self.idx_off = 0
        self.idx_one = 2
        self.idx_bh_start = 4
        self.idx_bh_end = 4       # will update
        self.step_cycles = 0

    def prepare(self):
        import math

        # Step (in sysclk cycles) for the BH segment only
        self.step_cycles = max(1, int(round(self.SYSCLK_HZ * self.BH_DWELL_S)))

        # Build RAM payload: [0.0, 1.0, BH-window...]
        amp = [0.0,0.0, 1.0,1.0]
        self.idx_off = 0
        self.idx_one = 2
        self.idx_bh_start = len(amp)

        # Blackman–Harris (4-term), ~8 us total
        n_bh = max(8, int(round(8e-6 / self.BH_DWELL_S)))
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        if n_bh < 4:
            # fallback raised-cosine if dwell is very coarse
            bh = [0.5 - 0.5*math.cos(2*math.pi*i/max(1, n_bh-1)) for i in range(n_bh)]
        else:
            twopi = 2*math.pi
            N = n_bh
            bh = []
            for n in range(N):
                x = n/(N-1)
                w = a0 - a1*math.cos(twopi*x) + a2*math.cos(2*twopi*x) - a3*math.cos(3*twopi*x)
                bh.append(w)
            # normalize to peak 1.0
            peak = max(bh)
            if peak > 0:
                bh = [v/peak for v in bh]
        amp.extend(bh)
        self.idx_bh_end = len(amp) - 1

        self.ram_amp = amp
        self.asf_words = [0]*len(amp)   # will be filled in kernel by amplitude_to_ram()

    @kernel
    def _program(self):
        self.core.reset()
        self.core.break_realtime()

        self.cpld.init()
        self.dds.init()

        self.core.break_realtime()

        self.dds.cfg_sw(True)
        self.dds.set_att(self.ATT)
        self.dds.set_frequency(self.CARRIER)

        self.core.break_realtime()

        # Convert floats → packed RAM words and write once
        self.dds.amplitude_to_ram(self.ram_amp, self.asf_words)
        self.dds.write_ram(self.asf_words)

        # ---- Set up profiles that point into the single RAM payload ----
        # P0: OFF (DIRECTSWITCH to single zero sample)
        self.dds.set_profile_ram(
            start=self.idx_off, end=self.idx_off, step=1,
            profile=0, mode=RAM_MODE_DIRECTSWITCH
        )

        # P1: SQUARE (DIRECTSWITCH to single one sample)
        self.dds.set_profile_ram(
            start=self.idx_one, end=self.idx_one, step=1,
            profile=1, mode=RAM_MODE_DIRECTSWITCH
        )

        # P2: BH envelope (one-shot RAMPUP over BH range)
        self.dds.set_profile_ram(
            start=self.idx_bh_start, end=self.idx_bh_end, step=self.step_cycles,
            profile=2, mode=RAM_MODE_RAMPUP
        )

        # Commit profile register writes
        # (Use any profile here; IO_UPDATE latches the profile config, not the pin selection.)
        self.cpld.set_profile(0)
        self.cpld.io_update.pulse_mu(8)

        # Enable RAM → ASF and latch
        self.dds.set_cfr1(ram_enable=1, ram_destination=RAM_DEST_ASF)
        self.cpld.io_update.pulse_mu(8)

        # RF on
        self.dds.sw.on()
        self.core.break_realtime()

    @kernel
    def _sequence(self):

        # Start from OFF
        self.cpld.set_profile(0)
        self.cpld.io_update.pulse_mu(8)
        delay(10*us)   # settle before the show (optional)

        self.cpld.set_profile(1)
        self.cpld.io_update.pulse_mu(8)
        delay(10*us)   # settle before the show (optional)

        self.cpld.set_profile(0)
        self.cpld.io_update.pulse_mu(8)
        delay(10*us)   # settle before the show (optional)

        """
        # Repeat: square 2us → off 1us → BH 8us → off 2us
        for _ in range(10):
            # Square ON duration controlled by ARTIQ (DIRECTSWITCH holds 1.0)
            self.cpld.set_profile(1)              # start square immediately (on next SYNC_CLK)
            delay(2*us)
            self.cpld.set_profile(0)              # OFF
            delay(1*us)

            # BH starts here; its *internal* duration is set by BH_DWELL_S * n_bh
            self.cpld.set_profile(2)              # triggers one-shot BH ramp
            delay(8*us)            # wait for BH to complete (matches how we built it)
            # (After end, AD9910 holds the last BH value, ~0; we switch to OFF anyway.)
            self.cpld.set_profile(0)
            delay(2*us)
        
        self.cpld.set_profile(0)
        delay(10*us)   # settle before the show (optional)

        # """

    @kernel
    def run(self):
        self._program()
        self._sequence()
