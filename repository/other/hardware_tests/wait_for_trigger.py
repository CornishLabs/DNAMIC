from artiq.experiment import *

class WaitForTtlTrigger(EnvExperiment):
    def build(self):
        self.setattr_device("core")
        self.setattr_device("ttl0")  # TTLInOut device in your device_db

    @kernel
    def wait_for_trigger(self):
        self.core.reset()
        self.ttl0.input()            # ensure it's in input mode
        self.core.break_realtime()   # avoid RTIO underflow on long waits
        self.ttl0.wait_edge()        # block until first edge (rising or falling)

    def run(self):
        # Block here until the trigger arrives
        self.wait_for_trigger()
        # Host-side print after kernel returns
        print("TRIGGERED")
