import glue_shower_functions as gsf

from param_card import *

print('Welcome to GlueShower v1.0\n')
print('Input Parameters:')
print('Nc = ' + str(Nc))
print('hadronization_scale_multiplier_c = ' + str(hadronization_scale_multiplier_c))
print('hadronization_temperature_multiplier_d = ' + str(hadronization_temperature_multiplier_d))
print('N_glueball_species_to_consider = ' + str(N_glueball_species_to_consider))
print('plasma_mode = ' + str(plasma_mode))
print('centre_of_mass_energy_gev = ' + str(centre_of_mass_energy_gev))
print('zero_pp_glueball_mass_gev = ' + str(zero_pp_glueball_mass_gev))
print('number_of_events = ' + str(number_of_events))
print('unconstrained_evolution = ' + str(unconstrained_evolution))
print('max_veto_counter = ' + str(max_veto_counter))
print('final_states_only = ' + str(final_states_only))
print('output_filename = ' + str(output_filename))
print('origin_particle_pid = ' + str(origin_particle_pid))

print('\nNow generating showers...')

gsf.file_output_showers(centre_of_mass_energy_gev,
	 zero_pp_glueball_mass_gev,
	 Nc,
	 hadronization_scale_multiplier_c,
	 hadronization_temperature_multiplier_d,
	 number_of_events,
	 plasma_mode,
	 N_glueball_species_to_consider,
	 output_filename,
	 final_states_only,
	 unconstrained_evolution,
	 max_veto_counter,
	 origin_particle_pid,
	 output_file_format)
