from artiq.experiment import *
from artiq.coredevice.ad9910 import (
    RAM_DEST_ASF,
    RAM_MODE_DIRECTSWITCH,
    RAM_MODE_RAMPUP,
    RAM_MODE_CONT_RAMPUP
)

class UrukulToneRAMExample(EnvExperiment):
    
    def build(self):
        self.setattr_device("core")
        self.dds = self.get_device("urukul4_ch0")
        self.cpld = self.get_device("urukul4_cpld")
        
    def prepare(self):
                   #  7  6    5    4     3    2    1    0     when writing ram profiles
        self.amp = [0.0, 0.0, 0.1, 0.7, 0.1, 0.5, 0.5, 0.0] # reverse order, as driver expects
        # If I add an extra leading zero to this, it breaks.
        self.asf_ram = [0] * len(self.amp)

    @kernel
    def init_dds(self, dds):
        self.core.break_realtime()
        dds.init()
        dds.set_att(6.*dB)
        dds.cfg_sw(True)

    @kernel
    def configure_ram_mode(self, dds):
        self.core.break_realtime()

        dds.set_cfr1(ram_enable=0) # Control Function Register 1
        self.cpld.io_update.pulse_mu(8) 

        # Enable the start RAM profile - I'm not sure why I *need* to do this here?
        # If I do it after write_ram, the DDS gives me an error and locks up
        self.cpld.set_profile(0)
        # Set the profile control register
        dds.set_profile_ram(
            start=0, end=0,
            step=250, profile=0, mode=RAM_MODE_DIRECTSWITCH
            )
        
        self.cpld.io_update.pulse_mu(8)

        self.cpld.set_profile(1) # You always need to set this before changing the RAM profile for some reason.
        dds.set_profile_ram(
            start=0, end=7,
            step=250, profile=1, mode=RAM_MODE_RAMPUP
            )
        
        self.cpld.io_update.pulse_mu(8)
        
        
        # Convert amp -> ram words (Amplitude Scale Factor (ASF))
        dds.amplitude_to_ram(self.amp, self.asf_ram)
        dds.write_ram(self.asf_ram)  # Write to RAM on the chip
        self.core.break_realtime() # Give time for write to happen
        
        
        dds.set(frequency=5*MHz, ram_destination=RAM_DEST_ASF) # Set what the frequency is, and what the RAM does (ASF)
        
        dds.set_cfr1(ram_enable=1, ram_destination=RAM_DEST_ASF) # Enable RAM, Pass osk_enable=1 to set_cfr1() if it is not an amplitude RAM
        self.cpld.io_update.pulse_mu(8) # Write to CPLD
    
    @kernel
    def run(self):
        self.core.reset()
        self.core.break_realtime()
        self.cpld.init()
        self.init_dds(self.dds)
        self.configure_ram_mode(self.dds)
        self.core.break_realtime()

        delay(6*us)
        self.cpld.set_profile(0)
        # self.cpld.io_update.pulse_mu(8)
        delay(6*us)
        self.cpld.set_profile(1)
        # self.cpld.io_update.pulse_mu(8)
        delay(4*us)
        # self.cpld.set_profile(1)
        # self.cpld.io_update.pulse_mu(8)
        # self.cpld.set_profile(0)
        # self.cpld.io_update.pulse_mu(8)
        # delay(20*us)
