# GlueShower v1.0
 
Software written by David Curtin, Caleb Gemmell, and Chris Verhaaren 

Stand alone generator found in folder 'GlueShower_v1', while example use and analysis (with pregenerated data) is in folder 'examples'

See paper "Simulating Glueball Production in $N_f = 0$ QCD" for more details, a subset of analysis performed in the paper is included in the 'examples' folder.
 
Please contact Caleb Gemmell (caleb.gemmell@mail.utoronto.ca / caleb.b.gemmell@gmail.com) with any issues/problems

# Contains:
- glue_shower_functions.py, Python file containing all required internal functions
- param_card.py, param file that is read by run_glue_shower.py
- run_glue_shower.py, Python script used to run GlueShower in command line (reads input parameters from param_card.py)
- inputs, various .csv files containing inputs related to the various SU(N) groups and glueballs
- lhe, output file for .lhe files
- dat, output file for .dat files (contains pre-generated sample runs)
- run_example.ipynb, example notebook of how to run GlueShower from python
- analysis_example.ipynb, example notebook of shower history analysis we are currently working on

# How to Run:
- GlueShower can either be run as a function in a Python script/notebook, as shown in example notebook "run_example.ipynb", or alternatively directly from the command line.
- To run in command line simply fill in the relevant parameters you want into the param_card.py file and once in the GlueShower directory on the command line simply run: "python run_glue_shower.py"

# Outputs
Outputs can either me formatted as a LHE file or a .dat file.
The .dat file is structured as follows, the file is a list of two entries, the first is a dictionary of parameters used to generate the shower histories, while the second is the shower histories themselves. The shower histories is a list of each event,
and each event contains a list of entries for each glueball (and gluon/plasmaball if included) with each entry containing the following information:
- Event ID (each particle assigned an ID for the event, e.g. 1 = first particle produced)
- Particle label (denotes which glueball state by the quantum numbers J^{PC}, or 'gluon' or 'gluon plasma')
- Particle mass (in units of GeV)
- Four momentum (in units of GeV)
- Parent particle ID (Event ID of particle that produced the current particle)
- Daughter particle ID (List of Event IDs of particles the current particle produces in the perturbative shower, empty list no perturbative daughters)


# Notes:
- Last tested on 23/02/22 using Python 3.7.1, on macOS (Big Sur 11.01)
- Need Numpy v1.21.0 or above
