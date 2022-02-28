########

# Glue shower functions

#######

import os
import math
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad, dblquad

############################################

# Alpha strong coupling as determined by the three loop beta function with zero flavours for general N
# as given in https://arxiv.org/pdf/hep-ph/0607209.pdf
def alpha_mirror_N(mu,mg,N,lambda_ratio):
    beta0 = (11/3)*N
    beta1 = (34/3)*N**2
    beta2 = 2857/2
    scale = mg/lambda_ratio
    L = 2*np.log(mu/scale)
    term2 = beta1*np.log(L)/(beta0**2 * L**2)
    term3 = beta1**2*(np.log(L)**2 - np.log(L) - 1)/(beta0**4 * L**3) + beta2/(beta0**3 * L**3)
    return (4*math.pi/beta0)*(1/L - term2 + term3)

############################################  
    
# Kinematic functions for a->bc splitting in lab frame
# Definition of splitting: Eb = z Ea, Ec = (1-z) Ea

############################################

# Daughter 4-momenta, in coords [E, pperp, plong], for given parent energy/3-momentum, daughter masses, splitting z
def atobc_pbsoln(Ea, pa, mb, mc, z):
    return [Ea*z,math.sqrt(-mb**2 + Ea**2*z**2 - (-mb**2 + mc**2 + pa**2 + Ea**2*(-1 + 2*z))**2/(4.*pa**2)),(-mb**2 + mc**2 + pa**2 + Ea**2*(-1 + 2*z))/(2.*pa)]
    
def atobc_pcsoln(Ea, pa, mb, mc, z):
    return [Ea - Ea*z,-math.sqrt(-mb**2 + Ea**2*z**2 - (-mb**2 + mc**2 + pa**2 + Ea**2*(-1 + 2*z))**2/(4.*pa**2)),(Ea**2 + mb**2 - mc**2 + pa**2 - 2*Ea**2*z)/(2.*pa)]

# Daughter opening angle
def costheta_daughters_general(Ea, ma, mb, mc, z):
    return (-ma**2 + mb**2 + mc**2 - 2*Ea**2*(-1 + z)*z)/(2.*math.sqrt(-mc**2 + Ea**2*(-1 + z)**2)*math.sqrt(-mb**2 + Ea**2*z**2))
    
# General zmin solution for arbitrary daughter masses
def zminmaxsoln_general (Ea, ma, mb, mc):
    return [(Ea*(ma**2 + mb**2 - mc**2) - Ea**2*math.sqrt((ma**4 + (mb**2 - mc**2)**2 - 2*ma**2*(mb**2 + mc**2))/(Ea**2 - ma**2)) + ma**2*math.sqrt((ma**4 + (mb**2 - mc**2)**2 - 2*ma**2*(mb**2 + mc**2))/(Ea**2 - ma**2)))/(2.*Ea*ma**2),(Ea*(ma**2 + mb**2 - mc**2) + Ea**2*math.sqrt((ma**4 + (mb**2 - mc**2)**2 - 2*ma**2*(mb**2 + mc**2))/(Ea**2 - ma**2)) - ma**2*math.sqrt((ma**4 + (mb**2 - mc**2)**2 - 2*ma**2*(mb**2 + mc**2))/(Ea**2 - ma**2)))/(2.*Ea*ma**2)]

# Zmin solution, assuming both daughters have m = mmin (used for evaluation of Sudakovs)
def zminsoln_mmin (Ea, ma, mmin):
    try:
        return 0.5 - math.sqrt((Ea**2 - ma**2)*(ma**2 - 4*mmin**2))/(2*Ea*ma)
    except ValueError:
        #is it that crazy numerical problem where sometimes qquad integration evaluates zmin at 10^-13 outside of tmin/max range?
        if (abs(ma - 2*mmin) < 10**-12):
            return zminsoln_mmin(Ea, 2*mmin, mmin)
        else:
            raise(ValueError)
        
# Special case: splitting of first two gluons
# Functions for z and t range
def zminsoln_firststep (M0, mmin, t):
    try:
        return (1 - math.sqrt(-(((4*mmin**2 - t)*(M0**4 + (mmin**2 - t)**2 - 2*M0**2*(mmin**2 + t)))/M0**2))/(math.sqrt(t)*math.sqrt((M0**2 - mmin**2 + t)**2/M0**2)))/2
    except ValueError:
        #is it that crazy numerical problem where sometimes qquad integration evaluates zmin at 10^-13 outside of tmin/max range?
        current_tminmax = tminmaxsoln_firststep(M0, mmin)
        if (abs(t - current_tminmax[0]) < 10**-10):
            return zminsoln_firststep(M0, mmin, current_tminmax[0])
        elif (abs(t - current_tminmax[1]) < 10**-10):
            return zminsoln_firststep(M0, mmin, currenttminmax[1])
        else:
            raise(ValueError)
        
def tminmaxsoln_firststep(M0, mmin):
    return [ 4*mmin**2,(M0 - mmin)**2 ]

# Finds updated z and t range once first two gluon virtualities have been assigned
def zminsoln_firststep_t1t2 (M0, mmin, t1, t2):
    return 0.5 - math.sqrt((-4*mmin**2 + t1)*(-math.sqrt(t1) + math.sqrt(t1 + (M0**4 + (t1 - t2)**2 - 2*M0**2*(t1 + t2))/(4.*M0**2)))*(math.sqrt(t1) + math.sqrt(t1 + (M0**4 + (t1 - t2)**2 - 2*M0**2*(t1 + t2))/(4.*M0**2))))/(2.*math.sqrt(t1)*math.sqrt(t1 + (M0**4 + (t1 - t2)**2 - 2*M0**2*(t1 + t2))/(4.*M0**2)))
    
def tminmaxsoln_firststep_t1t2(M0, mmin, t2):
    return [4*mmin**2,M0**2 - 2*M0*math.sqrt(t2) + t2]
    
# p1solnfirststep and p2solnfirststep just give gluon 1, 2, momenta assuming the gluons have virtualities t1, t2
def p1soln_firststep(M0, t1, t2):
    return [math.sqrt(t1 + (M0**4 + (t1 - t2)**2 - 2*M0**2*(t1 + t2))/(4.*M0**2)),0,0,math.sqrt(M0**4 + (t1 - t2)**2 - 2*M0**2*(t1 + t2))/(2.*M0)]
        
def p2soln_firststep(M0, t1, t2):
    return [math.sqrt(t2 + (M0**4 + (t1 - t2)**2 - 2*M0**2*(t1 + t2))/(4.*M0**2)),0,0,-math.sqrt(M0**4 + (t1 - t2)**2 - 2*M0**2*(t1 + t2))/(2.*M0)]

# Finds updated z value once daughters have been assigned actual virtualities
def z_new (ma, mb, mc, zi, mmin):
    return ((ma**2 + mb**2 - mc**2)/ma + (-1 + 2*zi)/math.sqrt((-4*mmin**2 + ma**2)/(ma**4 + (mb**2 - mc**2)**2 - 2*ma**2*(mb**2 + mc**2))))/(2.*ma)

############################################  
    
# Sudakov functions

############################################

# Gluon-to-gluon splitting function    
def Phatgg(z,N):
    return 2*N*(z/(1 - z) + (1 - z)/z + z *(1 - z))

# Sudakov integrand
def sudakov_integrand(t, z, m0, N, lambda_ratio):
    return -(2*t*math.pi)**(-1) * Phatgg(z,N) * alpha_mirror_N(math.sqrt(z*(1-z)*t), m0, N, lambda_ratio)

# Log of the sudakov
def LogSudakov(t, tmin, m0, N, zminfn, lambda_ratio):
    return dblquad(lambda z, tprime: sudakov_integrand(tprime, z, m0, N, lambda_ratio), tmin, t, lambda tprime: zminfn(tprime), lambda tprime: 1- zminfn(tprime) )[0]

# Actual sudakov
def Sudakov(t, tmin, m0, N, zminfn, lambda_ratio):
    return math.exp(LogSudakov(t, tmin, m0, N, zminfn, lambda_ratio))

# Needed for following function
def return_element_with_closest_value(input_list, comparison_value, position = -1):
    if(position == -1): # we actually have a 1D list
        temp = [[abs(item - comparison_value), item] for item in input_list]
        temp.sort()
        return temp[0][1]
    else:
        temp = [[abs(item[position] - comparison_value),item] for item in input_list]
        temp.sort()
        return temp[0][1]

# This wrapper class for the log_Sudakov evaluation remembers every tmin, tmax value it 's been evaluated for, 
# and only integrates over the t-ranges it doesn't already know to return sudakov for the desired t-values. 
# speeds up evaluation by about factor of 5.
class efficient_log_Sudakov:
    def __init__(self, tmin_input, m0_input, N_input, zminfn_input, lambda_ratio):
        self.logtlogsudakovList =[]
        self.tmin = tmin_input
        self.m0 = m0_input
        self.zminfn = zminfn_input
        self.N = N_input
        self.lambda_ratio = lambda_ratio
        
        self.current_logSudakov_fn = lambda logt1, logt2: LogSudakov(math.exp(logt2), math.exp(logt1), self.m0, self.N, self.zminfn, self.lambda_ratio)
        
        self.logtlogsudakovList.append([math.log(tmin_input), self.current_logSudakov_fn(math.log(tmin_input),math.log(tmin_input))])
        
    def eval(self, logt):
        closest_logtlogsudakov = return_element_with_closest_value(self.logtlogsudakovList, logt, 0)
        if (closest_logtlogsudakov[0] == logt):
            return closest_logtlogsudakov[1] # if we already evaluated this logt-value
        else:
            new_logsudakov = closest_logtlogsudakov[1] + self.current_logSudakov_fn(closest_logtlogsudakov[0], logt)
            self.logtlogsudakovList.append([logt, new_logsudakov])
            self.logtlogsudakovList.sort()
            return new_logsudakov

# Finds a value of t for the gluon, following the Monte Carlo process outlined in the pink book
def find_t_split(randomR, tstart, tmin, m0, N, zminfn, lambda_ratio):
    currenttsplit = -1 # output if gluon does not split
    
    # use our fancy new efficient sudakov, that's particularly good if you start at tmin and work your way up
    current_logSudakov_fn_object = efficient_log_Sudakov(tmin, m0, N, zminfn, lambda_ratio)
    current_logSudakov_fn = lambda logt: current_logSudakov_fn_object.eval(logt)
        
    
    # we want to find t such that Sudakov(t, tmin, m0, zminfn) = desired_Sudakov
    desired_logSudakov = current_logSudakov_fn(math.log(tstart)) - math.log(randomR);
    
    if desired_logSudakov < 0:
        current_logt = brentq(lambda logt : current_logSudakov_fn(logt) - desired_logSudakov, math.log(tmin), math.log(tstart),
                             xtol=1e-6)

        currenttsplit = math.exp(current_logt)
        
    del current_logSudakov_fn_object
    return currenttsplit
    
# Finds a value of z for the gluon, following the Monte Carlo process outlined in the pink book
def find_z_split(randomR, tsplit, m0, N, zmin, lambda_ratio):
    desired_unrenormalized_chancesplit = randomR * quad(lambda z: sudakov_integrand(tsplit, z, m0, N, lambda_ratio), zmin, 1-zmin)[0]
    
    current_unrenormalized_chancesplit_fn = lambda zsplit: quad(lambda z: sudakov_integrand(tsplit, z, m0, N, lambda_ratio), zmin,
                                                                          zsplit)[0]
    
    currentzsplit = brentq(lambda z : current_unrenormalized_chancesplit_fn(z) - desired_unrenormalized_chancesplit,
                           zmin, 1-zmin,xtol=1e-6)
    
    return currentzsplit
    
############################################  
    
# Glueball functions

############################################

# Finds the critical temperature of the confining transition using https://arxiv.org/pdf/1202.6684.pdf
# and puts in terms of lambda using https://arxiv.org/pdf/2106.00364.pdf
def temp_scale(N,string_tension):
    return (0.5949 + 0.458/N**2) * string_tension

# Finds relative probability of forming the glueballs, assuming a thermal distribution as used in https://arxiv.org/pdf/0908.1790.pdf
# NOTE: temp_scale is relative to the hadronisation scale (e.g. 2 = 2*had_scale)
def glueball_prob_distribution(N,temp_param,num_of_glueballs,glueball_mass_ratio,k,glueballs,string_tension):
    relative_multiplicities = [ (2*float(glueballs[i][0]) + 1)*(glueball_mass_ratio[i]**1.5)*math.e**(-k*(glueball_mass_ratio[i] - 1)/(temp_param*temp_scale(N,string_tension))) for i in range(0,num_of_glueballs)]
    return np.array(relative_multiplicities)/np.sum(relative_multiplicities)
    
# Hadronise function takes the current virtuality of the gluon, and picks from a weighted function of available glueball states,
# returning the mass squared.
def Hadronise(currentt,glueball_masses,glueball_probs):
    currentmass = math.sqrt(currentt)
    if currentmass >= max(glueball_masses):
        cutoff_pos = len(glueball_masses)
    elif currentmass < glueball_masses[0]:
        cutoff_pos = 1
    else:
        cutoff_pos = next(x[0] for x in enumerate(glueball_masses) if x[1] > currentmass)
    current_glueball_probs = np.array(glueball_probs[0:cutoff_pos])/sum(glueball_probs[0:cutoff_pos])
    return glueball_masses[np.random.choice(range(len(current_glueball_probs)), p = current_glueball_probs)]**2

# Takes a mass and returns the glueball label with the equivalent mass
def get_glueball_type(mass,glueball_masses,glueballs):
    return glueballs[ np.where(np.round(glueball_masses,1) == np.round(mass,1))[0][0] ]
    
############################################  
    
# Gluon plasma functions (follows hep-ph:1612.00850 and arxiv:1305.5226)

############################################
    
# Simply boosts a four vector
def BoostVector(four_vector,boost,direction):
    gamma = 1/math.sqrt(1-boost**2)
    # Ensure direction is unit vector
    direction = np.array(direction)/math.sqrt(np.dot(np.array(direction), np.array(direction)))
    dot = np.dot(four_vector[1:4],direction)
    E_boost = gamma*( four_vector[0] - boost*dot)
    px_boost = four_vector[1] + ((gamma - 1)*dot - gamma*boost*four_vector[0])*direction[0]
    py_boost = four_vector[2] + ((gamma - 1)*dot - gamma*boost*four_vector[0])*direction[1]
    pz_boost = four_vector[3] + ((gamma - 1)*dot - gamma*boost*four_vector[0])*direction[2]
    return np.array([E_boost,px_boost,py_boost,pz_boost])

# Maxwell-boltzmann dist as used in arxiv:1305.5226    
def dist(p,A):
    return p**2*math.exp(-A*p**2/(1+math.sqrt(1 + p**2)))

# Derivative of maxwell-boltzmann dist
def dist_deriv(p,A):
    return math.exp(-A*p**2/(1+math.sqrt(1 + p**2)))*p*(2 - A*p**2/math.sqrt(1+p**2))

# Algorithm to randomly sample from a MB dist, listed in arxiv:1305.5226   
def get_momentum_mag(m,T):
    A = m/T
    p_max = math.sqrt(2 + 2*math.sqrt(1 + A**2))/A
    p_mode = math.sqrt(2*(1 + np.sqrt(1 + A**2))/A**2)
    p_minus = brentq(lambda p: dist(p,A) -  dist(p_mode,A)/math.e, 10**-10, p_max)
    p_plus = brentq(lambda p: dist(p,A) -  dist(p_mode,A)/math.e, p_max, 100)
    lambda_minus = dist(p_minus,A)/dist_deriv(p_minus,A)
    lambda_plus = - dist(p_plus,A)/dist_deriv(p_plus,A)
    q_minus = lambda_minus/(p_plus - p_minus)
    q_plus = lambda_plus/(p_plus - p_minus)
    q_mode = 1 - (q_plus + q_minus)
    
    loop = True
    mom = None
    
    while loop:
        U = np.random.rand()
        V = np.random.rand()
        
        if U <= q_mode:
            Y = U/q_mode
            X =  (1 - Y)*(p_minus + lambda_minus) + Y*(p_plus - lambda_plus)
            if V <= dist(X,A)/dist(p_mode,A) and X > 0:
                loop = False
                
        elif U <= (q_mode + q_plus):
            E = - math.log((U - q_mode)/q_plus)
            X = p_plus - lambda_plus*(1 - E)
            if V <= math.exp(E)*dist(X,A)/dist(p_mode,A) and X > 0:
                loop = False
                
        else:
            E = - math.log((U - (q_mode + q_plus))/q_minus)
            X = p_minus + lambda_minus*(1 - E)
            if V <= math.exp(E)*dist(X,A)/dist(p_mode,A) and X > 0:
                loop = False
    
    return X*m
   
# Decays plasma and boosts daughters to lab frame
def decay_plasma(four_momentum,m,glueball_masses,glueball_probs,N,string_tension,d):
    T = d*temp_scale(N,string_tension)
    
    # Rescale 4vectors to conserve energy
    def rescale_func(x,glueballs,Etot):
        E = []
        for i in glueballs:
            E.append(math.sqrt(i[0]*i[0] + (x*x - 1)*np.dot(i[1:],i[1:])))
        return sum(E) - Etot
    
    test = True
    while test:
        
        # Get list of 4vectors for thermally emitted glueballs until mass 'runs out'
        glueball_4moms = []
        m_counter = m
        
        while m_counter > 0:

            glueball_mass = np.round(glueball_masses[np.random.choice(range(len(glueball_probs)), p = glueball_probs)],1)
            
            if m_counter - glueball_mass > 0:
                m_counter = m_counter - glueball_mass

                mom_mag = get_momentum_mag(glueball_mass,T)
                phi = 2*math.pi*np.random.rand()
                theta = math.acos(2*np.random.rand() - 1)
                glueball_4moms.append(np.array([math.sqrt(mom_mag**2 + glueball_mass**2),mom_mag*math.cos(phi)*math.sin(theta),
                            mom_mag*math.sin(phi)*math.sin(theta),mom_mag*math.cos(theta)]))
            else:
                m_counter = 0

        if len(glueball_4moms) >= 2:    
            # Boost 4vectors to C.O.M. frame
            unbalanced_mom = np.array(glueball_4moms).sum(axis=0)

            balance_boost = math.sqrt(np.dot(unbalanced_mom[1:],unbalanced_mom[1:]))/unbalanced_mom[0]
            balance_boosted_glueballs = [ BoostVector(i,balance_boost,unbalanced_mom[1:]) for i in glueball_4moms ]
            
            if rescale_func(0,balance_boosted_glueballs,m) < 0 and rescale_func(10,balance_boosted_glueballs,m) > 0:
                test = False
    
    a = brentq(lambda x: rescale_func(x,balance_boosted_glueballs, m), 0, 10)
    
    rescaled_glueballs = [ np.array([math.sqrt(i[0]*i[0] + (a*a - 1)*np.dot(i[1:],i[1:])),
                                     a*i[1],a*i[2],a*i[3]]) for i in balance_boosted_glueballs]
    
    # Boost glueballs to frame of initial gluon emission
    final_boost = math.sqrt(np.dot(four_momentum[1:],four_momentum[1:]))/four_momentum[0]
    
    if final_boost == 0.0:
        final_boosted_glueballs = rescaled_glueballs
    else:
        final_boosted_glueballs = [ BoostVector(i,-final_boost,four_momentum[1:]) for i in rescaled_glueballs ]
    
    return [ [ np.round(math.sqrt(i[0]**2 - np.dot(i[1:],i[1:])),1), i.tolist() ] for i in final_boosted_glueballs ]
    
############################################  
    
# Actual main shower functions

############################################

# Easy labels for referring to entries of a single shower history
SHlabel = 1 - 1
SHm = 2 - 1
SHp = 3 - 1
SHzi = 4 - 1
SHz = 5 - 1
SHthetamax = 6 - 1
SHparentlabel = 7 - 1
SHdaughterlabels = 8 - 1
SHtype = 9 - 1 

def new_SH_entry():
    return [-1,-1,-1,-1,-1,-1,-1,-1,-1]
    
# Return the position within showerhistory where a given parton is
def parton_index(showerhistory, partonlabel):
    parton = [item for item in showerhistory if item[SHlabel] == partonlabel][0]
    return showerhistory.index(parton)

# Starts the shower, initialises the z,t values of initial two gluons, further splittings handled by following function
def Generate_Gluon_Shower(M0, m0, N, glueball_masses, glueball_probs, lambda_ratio, glueballs, c = 1,
                          gluon_plasma_option = False, unconstrained_evolution = True, max_veto_counter = 100):
    # global vars that the function defines have to be defined before setting
    
    
    # GLOBAL VARS: will be reset each time you run Generate_Gluon_Shower
    global Shower_History # really the only global variable should be the shower history. just easier that way. 
    global next_SH_label # keeps track of the label we assign to the next parton.
        
    # had_scale sets branching scale, c allows scaling to value different than glueball mass
    # mmin is just the minimum mass a gluon can evole to.
    had_scale = c*(2*m0)
    mmin = had_scale/2
    
    # reset global shower history variables
    Shower_History = []
    next_SH_label = 1
        
    # current tmin/max variables
    tmin_firststep, tmax_firststep = tminmaxsoln_firststep(M0,mmin)
    
    
    def evolve_gluon (tstart):
        hadronise = False

        if (tstart < had_scale**2):
            hadronise = True
        else:
            tguess = find_t_split(np.random.rand(), tstart, tmin_firststep, m0, N,
                                  lambda t: zminsoln_firststep(M0, mmin, t), lambda_ratio)

        if not hadronise:
            if (tguess > had_scale**2):  
                zguess = find_z_split(np.random.rand(), tguess, m0, N,
                                      zminsoln_firststep(M0, mmin, tguess), lambda_ratio)
            else:
                hadronise = True
        if gluon_plasma_option:
            return [ mmin**2, -1] if hadronise else [tguess, zguess]
        else:
            return [Hadronise(tstart,glueball_masses,glueball_probs), -1] if hadronise else [tguess, zguess]
       
    #assuming the other gluon has m=mmin, find first guess for each gluon t1,2, and using that t1,2. find z1,2
    t1guess,z1guess = evolve_gluon(tmax_firststep)
    t2guess,z2guess = evolve_gluon(tmax_firststep)
    
    # APPLY VETO: 
    # Check if the splittings we computed for gluon1,2 are consistent with each other. 
    # If not, evolve gluon further. Do this at most max_veto_counter times.

    def zguessOK(zguess,t1guess,t2guess):
        if math.sqrt(t1guess) + math.sqrt(t2guess) > M0:
            return False
        elif (zguess == -1):
            return True
        else:
            try:
                actual_zmin = zminsoln_firststep_t1t2 (M0, mmin, t1guess, t2guess)
                return (actual_zmin < zguess < 1 - actual_zmin)
            except ValueError:
                # zminsoln_firststep_t1t2 gave ValueError, i.e. imaginary part, i.e. z is NOT OK
                return False
    
    
    counter = 0
    while (counter <= max_veto_counter and (zguessOK(z1guess,t1guess,t2guess) == False or
                                           zguessOK(z2guess,t2guess,t1guess) == False)):
        counter+=1
        
        # z1 not OK: evolve gluon1 down
        if (zguessOK(z1guess,t1guess,t2guess) == False and zguessOK(z2guess,t2guess,t1guess) == True):
            t1guess,z1guess = evolve_gluon(t1guess)
        # z2 not OK: evolve gluon2 down
        elif (zguessOK(z1guess,t1guess,t2guess) == True and zguessOK(z2guess,t2guess,t1guess) == False):
            t2guess, z2guess = evolve_gluon(t2guess)
        # if both not OK, which also happens when gluon1,2 exceed mass of M0, evolve down gluon with larger mass
        elif (zguessOK(z1guess,t1guess,t2guess) == False and zguessOK(z2guess,t2guess,t1guess) == False):
            if t1guess > t2guess:
                t1guess, z1guess = evolve_gluon(t1guess)
            else:
                t2guess, z2guess = evolve_gluon(t2guess)
    
    
    if (counter > max_veto_counter):
        print("First splitting could not resolve. Aborting.")
        return None
    else:
              
        parton1 = new_SH_entry()
        parton2 = new_SH_entry()
        
        parton1[SHlabel] = next_SH_label
        next_SH_label+=1
        parton2[SHlabel] = next_SH_label
        next_SH_label+=1
        
        parton1[SHm] = math.sqrt(t1guess)
        parton2[SHm] = math.sqrt(t2guess)
        parton1[SHzi] = z1guess
        parton2[SHzi] = z2guess
        
        parton1[SHp] = p1soln_firststep(M0, t1guess, t2guess)
        parton2[SHp] = p2soln_firststep(M0, t1guess, t2guess)
        
        parton1[SHparentlabel] = 0
        parton2[SHparentlabel] = 0
        
        parton1[SHthetamax] = math.pi
        parton2[SHthetamax] = math.pi
        
        if(parton1[SHzi] > 0):
            parton1[SHdaughterlabels] = [next_SH_label,next_SH_label+1]
            next_SH_label=next_SH_label+2
            parton1[SHtype] = 'gluon'
        else:
            parton1[SHdaughterlabels] = []
            parton1[SHz] = -1
            if gluon_plasma_option:
                parton1[SHtype] = 'gluon plasma'
            else:
                parton1[SHtype] = get_glueball_type(parton1[SHm],glueball_masses,glueballs)
        
        if(parton2[SHzi] > 0):
            parton2[SHdaughterlabels] = [next_SH_label,next_SH_label+1]
            next_SH_label=next_SH_label+2
            parton2[SHtype] = 'gluon'
        else:
            parton2[SHdaughterlabels] = []
            parton2[SHz] = -1
            if gluon_plasma_option:
                parton2[SHtype] = 'gluon plasma'
            else:
                parton2[SHtype] = get_glueball_type(parton2[SHm],glueball_masses,glueballs)
        
        Shower_History.append(parton1)
        Shower_History.append(parton2)
        
        Evolve_Daughters(parton1, m0, N, glueball_probs, glueball_masses, lambda_ratio, glueballs, c, gluon_plasma_option, unconstrained_evolution, max_veto_counter)
        Evolve_Daughters(parton2, m0, N, glueball_probs, glueball_masses, lambda_ratio, glueballs, c, gluon_plasma_option, unconstrained_evolution, max_veto_counter)
        
        return Shower_History
 
 
# Handles the successive splittings of the gluon shower     
def Evolve_Daughters(parent, m0, N, glueball_probs, glueball_masses, lambda_ratio, glueballs,
                     c, gluon_plasma_option, unconstrained_evolution, max_veto_counter):
    
    # GLOBAL VARS: will be reset by main function Generate_Gluon_Shower each time you run Generate_Gluon_Shower
    global Shower_History # really the only global variable should be the shower history. just easier that way. 
    global next_SH_label # keeps track of the label we assign to the next parton.
    
    # had_scale sets branching scale, c allows scaling to value different than glueball mass
    # mmin is just the minimum mass a gluon can evole to.
    had_scale = c*(2*m0)
    mmin = had_scale/2
    
    
    # ok let's go
    
    # for easy vector math, use numpy arrays. later when saving daughter momenta to Shower_History, 
    # convert to simple lists for easy saving etc
    
    parentm = parent[SHm]
    parentp = np.array(parent[SHp])
    parentzi = parent[SHzi]
    parentz = -1
    parentlabel = parent[SHlabel]
    current_theta_max = parent[SHthetamax]
    
    if (parentzi > 0):
        
        # only do anything if the parent actually splits
        
        daughter1 = new_SH_entry()
        daughter2 = new_SH_entry()
        
        daughter1label = parent[SHdaughterlabels][0]
        daughter2label = parent[SHdaughterlabels][1]
        
        # set up the coordinate system in accordance with a -> bc kinematics. 
        # "z" is direction of parent momentum, "x,y" span perpendicular plane
        
        zunitvector = parentp[1:4]
        zunitvector = zunitvector * 1/math.sqrt(np.dot(zunitvector, zunitvector))
        if ( (zunitvector == np.array([1,0,0])).all() ): 
            #extremely unlikely but just to be sure we don't get xunitvector = 0
            xunitvector = np.cross(np.array([0,1,0]), zunitvector)
        else:
            xunitvector = np.cross(np.array([1,0,0]), zunitvector)
        xunitvector = xunitvector * 1/math.sqrt(np.dot(xunitvector, xunitvector))
        yunitvector = np.cross(zunitvector, xunitvector)
        yunitvector = yunitvector * 1/math.sqrt(np.dot(yunitvector, yunitvector))
        
        # choose a random angle in xy plane to define the p_perp direction of the a->bc splitting
        randomangle = np.random.rand()*2*math.pi
        perp_unitvector = math.sin(randomangle)*xunitvector + math.cos(randomangle)*yunitvector
        
        
        # set up the Ea, Eb, etc values to use in the defined atobc kinematics
        Ea = parentp[0]
        pa = math.sqrt(np.dot(parentp[1:4], parentp[1:4]))
        ma = parentm
        
        
        Eb = parentzi * Ea
        Ec = (1-parentzi) * Ea
        
        # just to initialize vars
        mb = mc = newEb = newEc = pdaughter1 = pdaughter2 = -1
        
        # we need to find the daughter's masses to evaluate pb, pc, i.e. the daughter's momenta. 
        # Simultaneously, we will find zb, zc, i.e. the daughter's splittings
        
        
        # define some internal functions we need
        
        def evolve_gluon (Egluon,tstart):
            hadronise = False

            if (tstart < had_scale**2):
                hadronise = True
            else:
                tguess = find_t_split(np.random.rand(),tstart,had_scale**2,m0, N,
                                      lambda t: zminsoln_mmin(Egluon, math.sqrt(t), mmin), lambda_ratio)

            if not hadronise:
                if (tguess > had_scale**2):  
                    zguess = find_z_split(np.random.rand(), tguess, m0, N,
                                          zminsoln_mmin(Egluon, math.sqrt(tguess), mmin), lambda_ratio)
                else:
                    hadronise = True

            if gluon_plasma_option:
                return [ mmin**2, -1] if hadronise else [tguess, zguess]
            else:
                return [Hadronise(tstart,glueball_masses,glueball_probs), -1] if hadronise else [tguess, zguess]
        
        
        
        def is_zguess_OK (zguess, Enew, m):
            if (zguess == -1):
                return True
            #elif Enew < m:
            #    return False
            else:
                try:
                    actual_zmin = zminsoln_mmin(Enew, m, mmin)
                    return (actual_zmin < zguess < 1 - actual_zmin)
                except ValueError:
                    # zminsoln_mmin gave ValueError, i.e. imaginary part, i.e. z is NOT OK
                    return False
        
        
        def is_parentz_OK_constrained_evolution (parentz, Ea, ma, mb, mc):
            try:
                zminmaxsoln_general_sub = zminmaxsoln_general(Ea, ma, mb, mc)
                return (mb + mc < ma 
                  and
                  zminmaxsoln_general_sub[0] < parentz < zminmaxsoln_general_sub[1]
                  )
            except ValueError:
                # zminmaxsoln_general_sub gave ValueError, i.e. imaginary part, i.e. z is NOT OK
                return False
        
        
        def daughter_4vectors(Ea, pa, mb, mc, parentz):
            atobc_pbsoln_subbed = atobc_pbsoln(Ea, pa, mb, mc, parentz)
            atobc_pcsoln_subbed = atobc_pcsoln(Ea, pa, mb, mc, parentz)
            
            pdaughter1 = np.concatenate((
                np.array([atobc_pbsoln_subbed[0]])
                ,
                atobc_pbsoln_subbed[1] * perp_unitvector + atobc_pbsoln_subbed[2] * zunitvector
            ))

            pdaughter2 = np.concatenate((
                np.array([atobc_pcsoln_subbed[0]])
                ,
                atobc_pcsoln_subbed[1] * perp_unitvector + atobc_pcsoln_subbed[2] * zunitvector
            ))
            
            return [pdaughter1,pdaughter2]


        
        # getting first t,z guesses for daughters
        # starting virtuality is min(daughter energy, parent mass), see pythia6 manual, top of p357
        tzbguess = evolve_gluon(Eb, min(Eb, ma)**2)
        tzcguess = evolve_gluon(Ec, min(Ec, ma)**2)
            # let's store t,z guesses for daughters b, c in tzguess lists instead of two vars, more compact code
            # tguess = tzguess[0], zguess = tzguess[1]
        
        
        def unconstrained_evolution():
            nonlocal counter, tzbguess, tzcguess, mb, mc, ma, parentz, newEb, newEc, pdaughter1, pdaughter2, max_veto_counter
            
            zguessOK = False
            
            while (zguessOK == False and counter <= max_veto_counter):
                counter+=1
                
                mb = math.sqrt(tzbguess[0])
                mc = math.sqrt(tzcguess[0])
                    
                #check that daughter masses OK
                if(mb + mc < ma):
                    # find updated parentz
                    
                    parentz = z_new(ma, mb, mc, parentzi, mmin)
                    newEb = parentz*Ea
                    newEc = (1-parentz)*Ea
                    
                    # check if one of zb, zc is now out of bounds with the updated parent z value.
                    # (only one can be out of bounds)
                    # if one of them is out-of-bounds, keep evolving that parton with ORIGINAL evolution conditions, 
                    # starting from tb,cguess, but using Eb,c instead of newEb,c as in zmin
                    
                    if (is_zguess_OK(tzbguess[1], newEb, mb) == False):
                        # evolve b
                        tzbguess = evolve_gluon(Eb, tzbguess[0])
                        
                    elif (is_zguess_OK(tzcguess[1], newEc, mc) == False):
                        # evolve c
                        tzcguess = evolve_gluon(Ec, tzcguess[0])
                        
                    else:
                        # zguess OK! construct daughter momenta and exit loop!
                        
                        zguessOK = True
                        mb = math.sqrt(tzbguess[0])
                        mc = math.sqrt(tzcguess[0])
                        pdaughter1pdaughter2 = daughter_4vectors(Ea, pa, mb, mc, parentz)
                        pdaughter1 =  pdaughter1pdaughter2[0]
                        pdaughter2 =  pdaughter1pdaughter2[1]
                        
                
                else:
                    # mb + mc is too big
                    if(mb > mc):
                        # evolve b
                        tzbguess = evolve_gluon(Eb, tzbguess[0])
                    else:
                        # evolve c
                        tzcguess = evolve_gluon(Ec, tzcguess[0])
                
        
        def constrained_evolution():
            nonlocal counter, tzbguess, tzcguess, mb, mc, ma, parentz, newEb, newEc, pdaughter1, pdaughter2, max_veto_counter
            
            parentz = parentzi
            zguessOK = False
            while (zguessOK == False and counter < max_veto_counter):
                counter+=1
                
                mb = math.sqrt(tzbguess[0])
                mc = math.sqrt(tzcguess[0])
                
                
                if(is_parentz_OK_constrained_evolution(parentz, Ea, ma, mb, mc)):
                    # zguess OK! construct daughter momenta and exit loop!
                    zguessOK = True
                    pdaughter1pdaughter2 = daughter_4vectors(Ea, pa, mb, mc, parentz)
                    pdaughter1 =  pdaughter1pdaughter2[0]
                    pdaughter2 =  pdaughter1pdaughter2[1]
                
                else:
                    if (mb > mc):
                        # evolve b
                        tzbguess = evolve_gluon(Eb, tzbguess[0])
                    else:
                        # evolve c
                        tzcguess = evolve_gluon(Ec, tzcguess[0])
            
        
        
        
        if (unconstrained_evolution):
            counter = 0
            unconstrained_evolution()
        else:
            counter = 0
            constrained_evolution()
            
    
        
        
        # DONE WITH INITIAL EVOLUTION
        
        
        if (counter > max_veto_counter):
            print("Daughter splitting could not resolve. Aborting.")
            return None
        else:
            
            # time to enforce angular ordering
            # here is the splitting angle for the parent that we just computed
            
            thetaa = math.acos(np.dot(pdaughter1[1:4], pdaughter2[1:4])/math.sqrt(np.dot(pdaughter1[1:4], pdaughter1[1:4])*np.dot(pdaughter2[1:4], pdaughter2[1:4])))
            
            # in this function we find splitting for daughters b, c of input parent a
            # we want to make sure that the future splitting angle of the daughters is LESS than the splitting 
            # in which the parent originated, which is stored in variable current_theta_max
            # A way of doing this which is analogous to the Pythia6 manual eqn 10.19 is to estimate the 
            # daughters' splitting angle by assuming that THEIR children have m = mmin, and if that is larger
            # than current_theta_max, then we say that daughter does not split and has to be evolved down further.
            # if both daughters have a splitting angle that's too large, then first evolve the one with larger
            # splitting angle down, re-compute 4-vectors, then check the other one
            
            # note that assuming the daughters split into gluons with mass m = mmin overestimates their splitting angle
            # .... i guess that's fine? standard practice? 
            
            passed_angular_ordering = False
            counter = 0
            
            while (passed_angular_ordering == False and counter <= max_veto_counter):
                counter+=1
                
                thetab_max = thetac_max = -1
                
                # in case t changed, define daughter masses again
                mb = math.sqrt(tzbguess[0])
                mc = math.sqrt(tzcguess[0])
                
                # for convenience define newEb, newEc again, so this is the current daughter energy
                # for both constrained and unconstrained evolution.
                newEb = parentz*Ea
                newEc = (1-parentz)*Ea
                
                if (tzbguess[1] != -1 and tzcguess[1] != -1):
                    #both daughters have splittings in their future
                    
                    thetab_max = math.acos(costheta_daughters_general(newEb, mb, mmin, mmin, tzbguess[1]))
                    thetac_max = math.acos(costheta_daughters_general(newEc, mc, mmin, mmin, tzcguess[1]))
                    
                    if (thetab_max < thetaa and thetac_max < thetaa):
                        passed_angular_ordering = True
                    else:
                        # If either daughter' s future splitting angle is larger than current 
                        # splitting angle, evolve the daughter with the largest future splitting angle 
                        # keep in mind that for unconstrained evolution, we evolve with original daughter E
                        if (thetab_max > thetac_max):
                            #evolve b
                            tzbguess = evolve_gluon(Eb, tzbguess[0])
                            mb = math.sqrt(tzbguess[0])
                        else:
                            # evolve c
                            tzcguess = evolve_gluon(Ec, tzcguess[0])
                            mc = math.sqrt(tzcguess[0])
                        
                
                elif(tzbguess[1] != -1 and tzcguess[1] == -1):
                    # only daughter b has a splitting in her future
                    #thetab_max = costheta_daughters_general(newEb, mb, mmin, mmin, tzbguess[1])
                    thetab_max = math.acos(costheta_daughters_general(newEb, mb, mmin, mmin, tzbguess[1]))
                    
                    if (thetab_max < thetaa):
                        passed_angular_ordering = True
                    else:
                        #evolve b
                        tzbguess = evolve_gluon(Eb, tzbguess[0])
                        mb = math.sqrt(tzbguess[0])
                            
            
                elif(tzbguess[1] == -1 and tzcguess[1] != -1):
                    # only daughter c has a splitting in her future
                    #thetac_max = costheta_daughters_general(newEc, mc, mmin, mmin, tzcguess[1])
                    thetac_max = math.acos(costheta_daughters_general(newEc, mc, mmin, mmin, tzcguess[1]))
                    
                    if (thetac_max < thetaa):
                        passed_angular_ordering = True
                    else:
                        #evolve c
                        tzcguess = evolve_gluon(Ec, tzcguess[0])
                        mc = math.sqrt(tzcguess[0])
                
                elif (tzbguess[1] == -1 and tzcguess[1] == -1):
                    passed_angular_ordering = True
                            
                
                if (unconstrained_evolution and passed_angular_ordering == False):
                    # if we evolved any daughter gluon, update new parent z. 
                    parentz = z_new(ma, mb, mc, parentzi, mmin)
                    newEb = parentz*Ea
                    newEc = (1-parentz)*Ea
                    mb = math.sqrt(tzbguess[0])
                    mc = math.sqrt(tzcguess[0])
                    
                    #pdaughter1pdaughter2 = daughter_4vectors(Ea, pa, mb, mc, parentz)
                    #pdaughter1 =  pdaughter1pdaughter2[0]
                    #pdaughter2 =  pdaughter1pdaughter2[1]

                    # restart angular evolution in case this gave z-values out of bounds (rare)
                    # note I did not do this in the mathematica code. one of the bugs there that apparently produced these error messages
                    if (is_zguess_OK(tzbguess[1], newEb, mb) and is_zguess_OK(tzcguess[1], newEc, mc)):
                        pdaughter1pdaughter2 = daughter_4vectors(Ea, pa, mb, mc, parentz)
                        pdaughter1 =  pdaughter1pdaughter2[0]
                        pdaughter2 =  pdaughter1pdaughter2[1]
                    else:
                        unconstrained_evolution()          
                
            
            # done with the angular ordering while loop
            
            if (counter > max_veto_counter):
                print("Daughter splitting could not resolve while imposing angular ordering. Aborting.")
                return None
            else:
                
                # WE ARE DONE. 
                
                # write updated final parentz to shower history
                Shower_History[parton_index(Shower_History, parentlabel)][SHz] = parentz
                
                # fill in daughter entries and add to shower history
                
                daughter1[SHlabel] = daughter1label
                daughter2[SHlabel] = daughter2label
                
                daughter1[SHm] = mb
                daughter2[SHm] = mc
                
                daughter1[SHp] = pdaughter1.tolist()
                daughter2[SHp] = pdaughter2.tolist()
                
                daughter1[SHzi] = tzbguess[1]
                daughter2[SHzi] = tzcguess[1]
                
                daughter1[SHthetamax] = daughter2[SHthetamax] = thetaa
                daughter1[SHparentlabel] = daughter2[SHparentlabel] = parentlabel
                
                if(daughter1[SHzi] > 0):
                    daughter1[SHdaughterlabels] = [next_SH_label,next_SH_label+1]
                    next_SH_label=next_SH_label+2
                    daughter1[SHtype] = 'gluon'
                else:
                    daughter1[SHdaughterlabels] = []
                    daughter1[SHz] = -1
                    if gluon_plasma_option:
                        daughter1[SHtype] = 'gluon plasma'
                    else:
                        daughter1[SHtype] = get_glueball_type(daughter1[SHm],glueball_masses,glueballs)

                if(daughter2[SHzi] > 0):
                    daughter2[SHdaughterlabels] = [next_SH_label,next_SH_label+1]
                    next_SH_label=next_SH_label+2
                    daughter2[SHtype] = 'gluon'
                else:
                    daughter2[SHdaughterlabels] = []
                    daughter2[SHz] = -1
                    if gluon_plasma_option:
                        daughter2[SHtype] = 'gluon plasma'
                    else:
                        daughter2[SHtype] = get_glueball_type(daughter2[SHm],glueball_masses,glueballs)
                    

                Shower_History.append(daughter1)
                Shower_History.append(daughter2)
                
                # evolve daughters
                
                Evolve_Daughters(daughter1, m0, N, glueball_probs, glueball_masses, lambda_ratio, glueballs, c, gluon_plasma_option, unconstrained_evolution, max_veto_counter)
                Evolve_Daughters(daughter2, m0, N, glueball_probs, glueball_masses, lambda_ratio, glueballs, c, gluon_plasma_option, unconstrained_evolution, max_veto_counter)

# THIS IS THE FUNCTION TO USE
# Allows multiple events to be simulated easily, also handles the decays of gluon plasma       
def OutputShowers(M0, m0, N, c, d, num_events, gluon_plasma_option, N_glueball_species_to_consider, final_states_only,
                  unconstrained_evolution, max_veto_counter, glueball_masses,glueballs,lambda_ratio,string_tension,glueball_probs):
    
    all_clean_shower_histories = []
    
    fails = 0
    successes = 0
    
    while successes != num_events:

        if gluon_plasma_option and c >= M0/(2*m0):                           
            initial_shower_history = [[1, M0, [ M0, 0, 0, 0], -1, -1, 0, 0, [], 'gluon plasma']]
        else:
            initial_shower_history = Generate_Gluon_Shower(M0, m0, N, glueball_masses, glueball_probs,
                                                           lambda_ratio, glueballs, c, gluon_plasma_option,
                                                           unconstrained_evolution, max_veto_counter)
        
        if initial_shower_history == None:
            fails += 1
        else:
            successes += 1
            
            if 100*successes/num_events % 5 == 0:
                print(str(int(100*successes/num_events)) + '% events generated')
            
            cleaned_shower = [ [i[0],i[-1],np.round(i[1],1),i[2],i[6],i[7]] for i in initial_shower_history ]
            showerID = max([ i[0] for i in cleaned_shower])

            for i in cleaned_shower:
                if i[1] == 'gluon plasma':
                    gluon_plasma_daughters = decay_plasma(i[3], float(i[2]), glueball_masses, glueball_probs, N,string_tension,d)
                    for j in gluon_plasma_daughters:
                        cleaned_shower.append([showerID+1,get_glueball_type(j[0],glueball_masses,glueballs),j[0],j[1],i[0]])
                        showerID += 1
                    
            if final_states_only:
                all_clean_shower_histories.append([ i for i in cleaned_shower if i[1] != 'gluon' and i[1] != 'gluon plasma'])
            else:
                all_clean_shower_histories.append(cleaned_shower)
                
    print("\nEvent generation completed. " + str(fails) + " events failed. " + str(successes) + " events successfully generated.\n")
    return all_clean_shower_histories

############################################  
    
# File output functions

############################################

def save_list(filename, list=-1):
    if(list == -1): # we use this function to save a globally accessible list by supplying its varname as a string
        listglobalvarname = filename
        with open(listglobalvarname+".dat", 'w') as file_handler: file_handler.write("{}\n".format(eval(listglobalvarname)))
    else:
        with open(filename, 'w') as file_handler: file_handler.write("{}\n".format(list))

# Essentially above function but writes to output files (dat or lhe)
def file_output_showers(M0, m0, Nc, c, d, number_of_events, gluon_plasma_option, N_glueball_species_to_consider = 12,
                        output_filename = 'default', final_states_only = False, unconstrained_evolution = True, max_veto_counter = 100,
                        origin_particle_pid = 25, output_file_format = 'dat'):
                        
    # Import relevant input files
    glueball_mass_ratios = np.genfromtxt('inputs/glueball_mass_ratios.csv',skip_header=3)
    glueball_masses = m0 * glueball_mass_ratios[np.where(glueball_mass_ratios == Nc)[0][0]][1:]
    glueballs = np.genfromtxt('inputs/glueball_mass_ratios.csv',skip_header=2, dtype=str)[0][1:]
    
    particle_ID_file = np.genfromtxt('inputs/particle_IDs.csv',skip_header = 1, delimiter = ',', dtype = 'str' )
    particle_IDs = {i[0]:int(i[1]) for i in particle_ID_file}
    
    if N_glueball_species_to_consider > len(glueball_masses) or  N_glueball_species_to_consider > len(particle_IDs):
        print("Number of glueballs requested is larger than number of glueballs listed in input files. Aborting...")
        print("Please include more information in glueball_mass_ratios.csv and/or particle_IDs.csv")
        return

    lambda_scales = np.genfromtxt('inputs/lambda_scales.csv',skip_header=3)
    lambda_ratio = lambda_scales[np.where(lambda_scales == Nc)[0][0]][1]
    
    string_tensions = np.genfromtxt('inputs/string_tensions.csv',skip_header=3)
    string_tension = string_tensions[np.where(string_tensions == Nc)[0][0]][1]
    
    glueball_probs = glueball_prob_distribution(Nc, d, N_glueball_species_to_consider,
                                                glueball_masses/m0,lambda_ratio,glueballs,string_tension)
                        
    # check param range
    if M0 < 2*m0:
        print("Initial mass can't split into two glueballs. Aborting...")
        return
    
    if gluon_plasma_option:
        if c <= 2:
            print("c has to be larger than 2 to run the gluon plasma option.")
            print("Turning gluon plasma option off.\n")
            gluon_plasma_option = False
            if c < 1:
                print("c must be greater than 1. Aborting ...")
                return
        if c >= M0/(2*m0):
            print('No splitting occurs, single gluon plasma decays...\n')
    else:
        if c < 1:
            print("c must be greater than 1. Aborting ...")
            return
            
    if final_states_only == True and output_file_format == 'LHE':
        print('LHE files require the full shower history. Setting final_states_only to False...\n')
        final_states_only = False
    
    events = OutputShowers(M0, m0, Nc, c, d, number_of_events, gluon_plasma_option, N_glueball_species_to_consider,
                                           final_states_only, unconstrained_evolution, max_veto_counter,
                                           glueball_masses,glueballs,lambda_ratio,string_tension,glueball_probs)
                                           
    shower_histories = [{'M':M0,'m0':m0,'N_c':Nc,'Lambda_had/2m_0':c,'T_had/T_c':d,'plasma-like':gluon_plasma_option,
    							'nGlueballs':N_glueball_species_to_consider,'nEvents':number_of_events,
    							'unconstrained evolution':unconstrained_evolution},events]
    
    if unconstrained_evolution:
    	unconstrained_label = 'unconstrained'
    else:
    	unconstrained_label = 'constrained'
    	
    c_label = str(c).replace('.','point')
    d_label = str(d).replace('.','point')
    
    if output_file_format == 'dat':
        if output_filename == 'default':
            filename = "dat/showerhistories_M_"+str(M0)+"_m0_"+str(m0)+"_N_"+str(Nc)+"_c_"+c_label+"_d_"+str(d_label)+"_plasma_"+str(gluon_plasma_option)+"_"+unconstrained_label+"_events_"+str(number_of_events)+".dat"
        else:
            filename = output_filename
        save_list(filename, shower_histories)
        print("Written to file: " + filename)
        return
        
    elif output_file_format == 'LHE':
    
        if output_filename == 'default':
            filename = "LHE/showerhistories_M_"+str(M0)+"_m0_"+str(m0)+"_N_"+str(Nc)+"_c_"+c_label+"_d_"+str(d_label)+"_plasma_"+str(gluon_plasma_option)+"_"+unconstrained_label+"_events_"+str(number_of_events)+".lhe"
        else:
            filename = output_filename

        try:
            os.remove(filename)
        except OSError:
            pass
    
        with open(filename, 'a') as file:
            file.write('<LesHouchesEvents>\n\n')
        
            file.write('<!--\nFILE GENERATED BY GLUESHOWER\n\n')
        
            file.write('Paramters:\n\nNc = '+str(Nc)+
                       '\nhadronisation_scale_multiplier_c = '+str(c)+
                       '\nthermal_probability_distribution_temperature_multiplier_d = '+str(d)+
                       '\nN_glueball_species_to_consider = '+str(N_glueball_species_to_consider)+
                       '\ngluon_plasma_option = '+str(gluon_plasma_option)+
                       '\ncentre_of_mass_energy_gev = '+str(M0)+
                       '\nzero_pp_glueball_mass_gev = '+str(m0)+
                       '\nnumber_of_events = '+str(number_of_events)+
                       '\nunconstrained_evolution = '+str(unconstrained_evolution)+
                       '\nmax_veto_counter = '+str(max_veto_counter)+
                       '\nfinal_states_only = '+str(final_states_only)+'\n')
                   
            file.write('-->\n\n')
        
            file.write('<init>\n')
                   
            file.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}'.format(origin_particle_pid,0,
            np.format_float_scientific(M0,6, min_digits = 6),
            np.format_float_scientific(0,6, min_digits = 6),
            0,0,247000,247000,-4,1))
        
            file.write('\n')
                   
            file.write("{0} {1} {2} {3}".format(np.format_float_scientific(7.624840e-06,6, min_digits = 6),
            np.format_float_scientific(1.381213e-08,6, min_digits = 6),
            np.format_float_scientific(7.624840e-06,6, min_digits = 6),1))
        
            file.write('\n')
                   
            file.write('</init>\n')
        
            for i in events:
                file.write('<event>\n')
                       
                file.write("{0:>2} {1:>6} {2} {3} {4} {5}".format(1 + len([x for x in i ]),1,
                np.format_float_scientific(1,7, min_digits = 7),
                np.format_float_scientific(-1,8, min_digits = 8),
                np.format_float_scientific(-1,8, min_digits = 8),
                np.format_float_scientific(-1,8, min_digits = 8)))
            
                file.write('\n')
                       
                file.write("{0:>9} {1:>2} {2:>4} {3:>4} {4:>4} {5:>4} {6} {7} {8} {9} {10} {11} {12:}".format(origin_particle_pid, -1, 0, 0, 0, 0,
                np.format_float_scientific(0,10, min_digits = 10,sign=True),
                np.format_float_scientific(0,10, min_digits = 10,sign=True),
                np.format_float_scientific(0,10, min_digits = 10,sign=True),
                np.format_float_scientific(M0,10, min_digits = 10),
                np.format_float_scientific(M0,10, min_digits = 10),
                np.format_float_scientific(0,10, min_digits = 4),
                np.format_float_scientific(0,10, min_digits = 4)))
            
                file.write('\n')
            
                for j in i:
                    if j[1] != 'gluon' and j[1] != 'gluon plasma':
                        file.write("{0:>9} {1:>2} {2:>4} {3:>4} {4:>4} {5:>4} {6} {7} {8} {9} {10} {11} {12:}".format(particle_IDs[j[1]], 1, j[4], 0, 0, 0,
                        np.format_float_scientific(j[3][1],10, min_digits = 10,sign=True),
                        np.format_float_scientific(j[3][2],10, min_digits = 10,sign=True),
                        np.format_float_scientific(j[3][3],10, min_digits = 10,sign=True),
                        np.format_float_scientific(j[3][0],10, min_digits = 10),
                        np.format_float_scientific(j[2],10, min_digits = 10),
                        np.format_float_scientific(0,10, min_digits = 4),
                        np.format_float_scientific(0,10, min_digits = 4)))
                    
                        file.write('\n')
                    else:
                        file.write("{0:>9} {1:>2} {2:>4} {3:>4} {4:>4} {5:>4} {6} {7} {8} {9} {10} {11} {12:}".format(particle_IDs[j[1]], 2, j[4], 0, 0, 0,
                        np.format_float_scientific(j[3][1],10, min_digits = 10,sign=True),
                        np.format_float_scientific(j[3][2],10, min_digits = 10,sign=True),
                        np.format_float_scientific(j[3][3],10, min_digits = 10,sign=True),
                        np.format_float_scientific(j[3][0],10, min_digits = 10),
                        np.format_float_scientific(j[2],10, min_digits = 10),
                        np.format_float_scientific(0,10, min_digits = 4),
                        np.format_float_scientific(0,10, min_digits = 4)))
                    
                        file.write('\n')

                file.write('</event>\n')
            
            file.write('</LesHouchesEvents>')
        
        print("Written to file: " + filename)
        return
        
    else:
        print("Please enter either \'LHE\' or \'dat\' as the output_file_format parameter in param_card.py")
        return