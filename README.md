# dnamic-lab
Durham Neutral Atom and Molecule Improved Control. Primarily Artiq code for experimental control, and NDSPs.

The wiki stores my questions, musings, and documentation for future people who go on this journey.

## How to use this repo

Get to the point of having a `nix develop` shell setup with https://github.com/CornishLabs/dnamic-setup.

Then:
```bash
cd ~/
git clone https://github.com/CornishLabs/dnamic-lab
cd dnamic-lab

# This script is only setup on the nix develop shell defined in the dnamic-setup repo
artiq-lab-tmux
```

This will launch a tmux session that starts all the necessary artiq components.
