######################################################################
####                                                              ####
####                  PARAM_CARD FOR GLUESHOWER                   ####
####                                                              ####
######################################################################



######################################################################

# SHOWER THEORY PARAMETERS

######################################################################

# Number of dark colours, SU(Nc), Nc = 2, 3, 4, 5, 6, 8, 10, 12 
Nc = 3

# Determines the scale at which the shower terminates, c * (2 * m0)
# (minimum value = 1)
hadronization_scale_multiplier_c = 1

# Temperature multiplier for the relative glueball multiplicity boltzmann distribution ~ Exp[m_i/(d * T_had)]
hadronization_temperature_multiplier_d = 1

# Number of glueball species you want to consider, ordered by increasing mass
N_glueball_species_to_consider = 12

# Gluon-plasma option determines the behaviour of the gluon at the hadronisation scale
# If False, gluon is turned into a single glueball (jet-like)
# If True, gluon becomes a gluon-plasma-ball with mass = c * m0, which then decays to glueballs
plasma_mode = True


######################################################################

# SHOWER GENERATION PARAMETERS

######################################################################

# Mass of the dark-colour-singlet particle that decays to two dark gluons
centre_of_mass_energy_gev = 100

# Lightest 0++ glueball mass m0
zero_pp_glueball_mass_gev = 10  

number_of_events = 100

# Unconstrained evolution is the default choice for Pythia
unconstrained_evolution = True

max_veto_counter = 100

# If True, only outputs the final glueballs
# If False, outputs the full shower history, including the dark gluons
final_states_only = False

# Option to specify file name, if 'default' given outputs filename of all input parameters
output_filename = 'default'

# Option to choose output file format
# 'LHE' = LHE file
# 'dat' = .dat file that can be read using the example analysis python notebook
output_file_format = 'LHE'

# If outputting to LHE file, allows you to input pID of dark colour singlet that decayed to two dark gluons
origin_particle_pid = 25










