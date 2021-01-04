import sys
import os
import h5py
import re
import psutil
import glob
import math
import numpy as np
#import numba
from time import time
import struct
from struct import pack
from struct import unpack, calcsize
from copy import deepcopy
import pygio

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
#import pydot
import pydotplus as pydot
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

default_sim = 'SV'

filenames = {'AQ':'',
             'SV':'m000p-',
             'HM':'',
            }

cc_template = {'AQ':'{}{}.corepropertiesextend.hdf5',
               'SV':'{}{}.corepropertiesextend.hdf5',
               'HM':'{}{}.coreproperties',
              } 
binfile_template = 'trees_099.{}'

mkeys = {'AQ':'m_evolved_0.9_0.005',
         'SV':'m_evolved_1.1_0.1',
         'HM':'m_evolved',
        }

# trees are built in 2 formats: 'core' and 'lgal'
core = 'core'
lgal = 'lgal'

# mode is serial or vector

# define properties from core catalog; also used as names for 'core' tree format
coretag = 'core_tag'
coremass = 'coremass'
infall_tree_node_mass = 'infall_tree_node_mass'
tree_node_mass = 'tree_node_mass'
fof_halo_mass = 'fof_halo_mass'
sod_halo_mass = 'sod_halo_mass'
sod_node_mass = 'sod_node_mass'
tree_node_index = 'tree_node_index'
fof_halo_tag = 'fof_halo_tag'
foftag = tree_node_index   #used to sort fof groups; uses halo fragments
infall_tree_node_index = 'infall_tree_node_index'
infall_fof_halo_mass = 'infall_fof_halo_mass'
infall_sod_halo_mass = 'infall_sod_halo_mass'
infall_sod_halo_cdelta = 'infall_sod_halo_cdelta'
infall_sod_halo_radius = 'infall_sod_halo_radius'
infall_fof_halo_angmom = 'infall_fof_halo_angmom_'
infall_sod_halo_angmom = 'infall_sod_halo_angmom_'

central = 'central'
timestep = 'timestep'
infall_step = 'infall_step'
radius = 'radius'

#define frequently used names for L-Galaxies
Descendent = 'Descendent'
FirstProgenitor = 'FirstProgenitor'
NextProgenitor = 'NextProgenitor'
FirstHaloInFOFGroup = 'FirstHaloInFOFGroup'
NextHaloInFOFGroup = 'NextHaloInFOFGroup'
MostBoundID_Coretag = 'MostBoundID->Coretag'
SubhaloIndex = 'SubhaloIndex'
ParentHaloTag = 'ParentHaloTag'
Offset = 'Offset'
DescendentOffset = Descendent + Offset
FirstProgenitorOffset = FirstProgenitor + Offset
NextProgenitorOffset = NextProgenitor + Offset
FirstHaloInFOFGroupOffset = FirstHaloInFOFGroup + Offset
NextHaloInFOFGroupOffset = NextHaloInFOFGroup + Offset
Len = 'Len'
SnapNum = 'SnapNum'
M_Crit200 = 'M_Crit200'
Zero = 'Zero'

first_snap = 499
last_snap = 43
#last_snap = 475
first_row = 0
last_row = 100 #101 snapshots between 499 and 43 
#last_row = 2

#properties available in core catalogs
#central, core_tag, fof_halo_tag, infall_sod_halo_mass, infall_fof_halo_angmom_x,y,z
#infall_fof_halo_mass, infall_step, infall_fof_halo_max_cir_vel, infall_fof_halo_tag,
#infall_sod_halo_angmom_x,y,z, infall_sod_halo_max_cir_vel, infall_tree_node_index,
#infall_tree_node_mass, tree_node_index, vel_disp, vx,y,z, x,y,z, m_evolved_*_*
#host_core?, radius, merged?

# properties for storing in dict or matrices
core_pointers = {lgal: [Descendent, DescendentOffset, FirstProgenitor, NextProgenitor,
                        FirstProgenitorOffset, NextProgenitorOffset],
                 core: [Descendent, FirstProgenitor, NextProgenitor]
                } 
#core_pointers = [Descendent, DescendentOffset, FirstProgenitor, FirstProgenitorOffset]

sibling_pointers = {lgal: [FirstHaloInFOFGroup, NextHaloInFOFGroup, FirstHaloInFOFGroupOffset, NextHaloInFOFGroupOffset],
                    core: [FirstHaloInFOFGroup, NextHaloInFOFGroup],
                   } 
#sibling_pointers = []

read_properties_float = {lgal: {'Pos_x':'x', 'Pos_y':'y', 'Pos_z':'z',
                                'Vel_x':'vx', 'Vel_y':'vy', 'Vel_z':'vz',
                                'VelDisp':'vel_disp',
                                'Vmax': 'infall_fof_halo_max_cir_vel',
                                M_Crit200: sod_node_mass,
                                'M_Mean200': Zero,
                                'M_TopHat': Zero,
                                'Spin_x': 'infall_sod_halo_angmom_x',
                                'Spin_y': 'infall_sod_halo_angmom_y',
                                'Spin_z': 'infall_sod_halo_angmom_z',
                                'SubHalfMass': Zero,
                               },
                         core: {'Pos_x':'x', 'Pos_y':'y', 'Pos_z':'z',
                                'Vel_x':'vx', 'Vel_y':'vy', 'Vel_z':'vz',
                                'VelDisp':'vel_disp',
                                'Vmax_fof': 'infall_fof_halo_max_cir_vel',
                                'Vmax_sod': 'infall_sod_halo_max_cir_vel',
                                infall_tree_node_mass: infall_tree_node_mass,
                                infall_fof_halo_mass: infall_fof_halo_mass,
                                infall_sod_halo_mass: infall_sod_halo_mass,
                                infall_sod_halo_cdelta: infall_sod_halo_cdelta,
                                infall_sod_halo_radius: infall_sod_halo_radius,
                                infall_fof_halo_angmom+'x': infall_sod_halo_angmom+'x',
                                infall_fof_halo_angmom+'y': infall_sod_halo_angmom+'y',
                                infall_fof_halo_angmom+'z': infall_sod_halo_angmom+'z',
                                coremass: coremass,
                                radius: radius,
                                tree_node_mass: tree_node_mass,
                                fof_halo_mass: fof_halo_mass,
                                sod_halo_mass: sod_halo_mass,
                                sod_node_mass: sod_node_mass,
                               },
                        } 

derived_properties_float = {lgal: [],
                            core: [],
                           } 

read_properties_int = {lgal: {MostBoundID_Coretag:coretag,
                              'FileNr': Zero,
                              SubhaloIndex: Zero,
                              fof_halo_tag: fof_halo_tag,
                             },
                       core: {coretag:coretag,
                              central: central,
                              fof_halo_tag: fof_halo_tag,
                              tree_node_index: tree_node_index,
                              infall_tree_node_index: infall_tree_node_index,
                              infall_step: infall_step,
                             },
                      } 

derived_properties_int = {lgal: [SnapNum, Len, ParentHaloTag],
                          core: [timestep],
                         }
properties_int32 = [SnapNum, timestep, infall_step, central]

def assemble_properties(fmt, vector=True):

    properties = {}
    properties['read'] = deepcopy(read_properties_float[fmt])   #need deepcopy so that update doesn't change original dict
    properties['read'].update(read_properties_int[fmt])
    properties['derived'] = derived_properties_int[fmt] + derived_properties_float[fmt]
    properties['float'] = derived_properties_float[fmt] + list(read_properties_float[fmt].keys())
    properties['int'] = core_pointers[fmt] + sibling_pointers[fmt] + derived_properties_int[fmt] + list(read_properties_int[fmt].keys())
    properties['pointers'] = core_pointers[fmt] + sibling_pointers[fmt]
        
    mode = 'vector' if vector else 'serial'
    print('Using {} format; assembling {} properties'.format(fmt, mode))
    print(properties)
    
    return properties

no_int = -999
no_float = -999.
no_mass = -101.
M_Crit_norm = 1.e10

#masses in #M_sun/h
particle_numbers = {'LJ':10752,
                    'SV':1024,
                    'HM':3072,
                    'AQ':1024,
                   }
box_size = {'LJ':3400,
            'SV':250,
            'HM':250,
            'AQ':256,
           }

planck_cdm = 0.26067
planck_wb = 0.02242
planck_h = .6766
wmap_cdm = .220
wmap_wb = 0.02258
wmap_h = 0.71
G_SI = 6.67430e-11 #SI m^3 kg^-1 s^-2
Msun = 1.989e30 #kg
Mpc = 3.086e22 #m
#G = 4.3e-9 #Mpc km^2 s^-2 M⊙^-1
G = G_SI*1e-6*Msun/Mpc
#rho_c = 3*1e4/(8.*np.pi*G)  #Msun/h Mpc^-3 h^3
rho_c = 2.77536627e11

Omega_m = {'LJ':planck_cdm + planck_wb/planck_h**2,
           'SV':planck_cdm + planck_wb/planck_h**2,
           'HM':planck_cdm + planck_wb/planck_h**2,
           'AQ':wmap_cdm + wmap_wb/wmap_h**2,
          }

# 8.6e8 Msun/h for Millennium I, 6.9e6 Msun/h for Millennium II
particle_masses = {'MTII':6.88e6,
                   'MTI':2.6e8,
                   'SV':1.3e9,
                   'LJ':2.7e9,
                   'AQ':1.2e9,
                  } 

def get_particle_mass(sim):
    if sim in Omega_m.keys():
        Om_0 = Omega_m[sim]
        mass = Om_0*rho_c*(float(box_size[sim])/float(particle_numbers[sim]))**3
        print('Using derived {} particle mass = {:.9e}\n'.format(sim, mass))
    else:
        mass = particle_masses[sim]
        print('Cannot compute mass exactly; using approximate assigned value for {} = {:.2e}\n'.format(sim, mass))
    return mass

def get_a(step, nsteps=500, z_in=200, a_fin=1):
    a_ini = (1./(1.+z_in))
    a = a_ini + (a_fin-a_ini)/nsteps * (step + 1)
    #z = 1./a - 1.

    return a


a_scaling = {lgal: {'VelDisp':1.,
                    'Vmax': 0.5,
                    'Spin_x': 2.,
                    'Spin_y': 2.,
                    'Spin_z': 2.,
                    },
             core: {},
            } 

# header file contains Ntrees, totNHalos, TreeNHalos
header_format = "<{}i"

struct_keys = [DescendentOffset, FirstProgenitorOffset, NextProgenitorOffset,
               FirstHaloInFOFGroupOffset, NextHaloInFOFGroupOffset,
               Len, 'M_Mean200', M_Crit200, 'M_TopHat',
               'Pos_x', 'Pos_y', 'Pos_z', 'Vel_x', 'Vel_y', 'Vel_z',
               'VelDisp', 'Vmax', 'Spin_x', 'Spin_y', 'Spin_z',
               MostBoundID_Coretag, SnapNum, 'FileNr',
               SubhaloIndex, 'SubHalfMass',
              ]
struct_format = "<iiiiiiffffffffffffffqiiif"
"""
#test
struct_keys = [DescendentOffset, FirstProgenitorOffset, NextProgenitorOffset,
               FirstHaloInFOFGroupOffset, NextHaloInFOFGroupOffset,
               Len, M_Crit200, MostBoundID_Coretag,
#               SnapNum, SubhaloIndex,
              ] 

#struct_format = "<iiiiiifqii"
struct_format = "<iiiiiifq"
"""

# cc = build_core_trees.get_core_snapshot( '../CoreCatalogs', snapshot)
def get_core_snapshot(coredir, snapshot, template=cc_template[default_sim], sim=default_sim):
    fn = os.path.join(coredir, template.format(filenames[sim], snapshot))
    data= {}
    if os.path.exists(fn):
        if sim == 'HM':
            coredata = pygio.read_genericio("fn")
        else:
            h5 = h5py.File(fn, 'r')
            coredata = h5['coredata']

        keys = [k for k in list(coredata.keys()) if 'm_evolved' not in k]    
        for k in keys + [mkeys[sim]]:
            data[k] = coredata[k][()]
    else:
        print('{} not found'.format(fn))

    return data

def add_mass_columns(corecat, sim=default_sim):
    mask = (corecat['central']==1) # get centrals
    central_mass = corecat[infall_tree_node_mass][mask] # get fof mass (possibly fragment)
    if mkeys[sim] in corecat.keys():
        corecat[coremass] = corecat[mkeys[sim]] # get evolved masses
        # the mass of centrals are not modeled by mass model 
        central_massmodel = corecat[coremass][mask]
        check_mass = np.count_nonzero(~np.isclose(central_mass, central_massmodel)) #check if any entries don't agree
        if check_mass > 0:
            print('Mass model != tree-node mass for central cores in {}/{} entries'.format(check_mass,
                                                                                           np.count_nonzero(mask))) 
        corecat[coremass][mask] = central_mass  #force mass for centrals = tree-node mass
    
    # now add tree_node masses and fof masses
    corecat = add_treenode_fof_sod_mass(corecat, mask)
        
    return corecat

def add_treenode_fof_sod_mass(corecat, mask):
    tree_index, inverse = np.unique(corecat[tree_node_index], return_inverse=True)
    assert np.sum(mask)==len(tree_index), 'number of unique halos not equal to number of centrals'

    #get order of tree_node_indexes for centrals (mask selects centrals)
    central_tree_index, cindex = np.unique(corecat[tree_node_index][mask], return_index=True)
    assert np.array_equal(tree_index, central_tree_index), 'Mismatch in tree_node_indexes of centrals'
    # use cindex to reorder central masses in (ascending) unique order
    for key in [tree_node_mass, fof_halo_mass, sod_halo_mass]:
        mass = corecat['infall_' + key][mask]
        unique_ordered_masses = mass[cindex]
        # reconstruct full array of central-only masses using inverse
        corecat[key] = unique_ordered_masses[inverse]
        print('Adding column {} to corecat'.format(key))
        #check
        assert np.array_equal(corecat[key][mask], corecat['infall_' + key][mask]), 'Mismatch in masses of centrals'
        
    #now assign sod masses if available; otherwise use treenode mass
    corecat[sod_node_mass] = corecat[tree_node_mass].copy()   #need to copy otherwise column getsa overwritten
    mask_sod = corecat[sod_halo_mass] > 0 #select valid sod masses
    nofrag_mask = corecat[fof_halo_tag] > 0
    mask_sod_nofrag = mask_sod & nofrag_mask
    mask_sod_frag = mask_sod & ~nofrag_mask
    #overwrite with valid sod masses that are not fragments
    corecat[sod_node_mass][mask_sod_nofrag] = corecat[sod_halo_mass][mask_sod_nofrag]
    #overwrite with valid sod masses multiplied by ratio of fragment mass to fof_halo mass
    corecat[sod_node_mass][mask_sod_frag] = corecat[sod_halo_mass][mask_sod_frag]*corecat[tree_node_mass][mask_sod_frag]/corecat[fof_halo_mass][mask_sod_frag]
    print('Adding column {} to corecat'.format(sod_node_mass))
    
    return corecat

def clean(corecat, sorted_coretags):
    mask = np.in1d(corecat[coretag], sorted_coretags, assume_unique=True)
    print('Truncating core catlog to {}/{} entries'.format(np.count_nonzero(mask),
                                                           len(corecat[coretag])))
    for p in list(corecat.keys()):
        corecat[p] = corecat[p][mask]

    return corecat

def add_snapshot_to_trees(coretrees, corecat, properties, current_snap, sim, snap_index=0,
                          print_int=100, coretags_fofm=None, argsorted_fofm=None,
                          sorted_indices=None, fmt=core,
                          ncore_min=0, ncore_max=None, vector=False): 

    assert coretags_fofm is not None, "No sibling-ordered tags supplied"
    assert sorted_indices is not None, "No sibling-ordered sorted indices supplied"
    #mass_order = np.flip(corecat[coremass].argsort())   #args in descending order

    # sort indices in this snapshot in decreasing foftag-mass order
    if current_snap < first_snap:
        sorted_indices_this = np.flip(np.lexsort(((corecat[coremass], corecat[foftag]))))
        coretags_fofm_this = corecat[coretag][sorted_indices_this]
    else:
        sorted_indices_this = sorted_indices
        coretags_fofm_this = coretags_fofm
    foftags_fofm_this = corecat[foftag][sorted_indices_this]  
        
    # get unique parents; index points to first occurence of tag; inverse reconstructs array
    parent_halos, index, inverse, counts = np.unique(foftags_fofm_this,
                                                     return_index=True, return_inverse=True,
                                                     return_counts=True)
    # initialize times
    stimes =  []
    atimes = []
    if vector:
        assert argsorted_fofm is not None, "No sibling-ordered argsorted indices supplied"
        # get required reordering of coretags in current snapshot for insertion into matrix
        # cannot assume cortetags will be in the same order as in last snapshot
        if current_snap < first_snap:
            # argsorted_fofm = coretags_fofm.argsort() 
            insertions = np.searchsorted(coretags_fofm[argsorted_fofm], coretags_fofm_this)
            locations_this = argsorted_fofm[insertions]
        else:
            locations_this = np.arange(0, len(coretags_fofm))
        print('Number of cores in snapshot {} (row {}) = {}'.format(current_snap, snap_index,
                                                                    len(locations_this)))
        # get siblings as vectors in sorted-indices order 
        stime1 = time()
        first_siblings, next_siblings = get_sibling_vectors(coretags_fofm_this, index, inverse)
        stimes.append((time()-stime1)/60.)

        ptime1 = time()
        # loop over property matrices in coretree dict
        for p, v in coretrees.items():     
            #define/reorder property
            atime1 = time()
            prop_values = get_ordered_property(p, corecat, sorted_indices_this,
                                               snap_index, current_snap, properties, 
                                               first_siblings, next_siblings, foftags_fofm_this,
                                               fmt=fmt, particle_mass=particle_masses[sim])
            atimes.append((time()-atime1)/60.)
            #fill row of property matrix with values for selected entries
            v[snap_index][locations_this] = prop_values
            #print('Sample value {}: {}'.format(p, v[snap_index][locations_this[0]]))
            
        # fix first progenitors
        coretrees[FirstProgenitor] = fix_firstprogenitor_vector(coretrees[FirstProgenitor], snap_index)
        if fmt == lgal:
            coretrees[FirstProgenitorOffset] = fix_firstprogenitor_vector(coretrees[FirstProgenitorOffset], snap_index)
        
        print('Time to fill coretree arrays = {:.2g} minutes'.format((time() - ptime1)/60.))

    else:  #serial code runs over parents of selected cores
        htime = time() 
        if not coretrees:  #first pass
            selected_cores = coretags_fofm_this[ncore_min:ncore_max]
            selected_parents = np.unique(foftags_fofm_this[ncore_min:ncore_max])
        else:
            #find parents belonging to selected cores being followed in tree dicts
            selected_cores = list(coretrees.keys())
            maskc = np.in1d(coretags_fofm_this, np.asarray(selected_cores)) #get mask for this coretag array
            selected_parents = np.unique(foftags_fofm_this[maskc])
        mask = np.in1d(parent_halos, selected_parents)
        # truncate parent halo arrays
        parent_halos = parent_halos[mask]
        counts = counts[mask]
        index = index[mask]
        if current_snap < first_snap:
            assert np.sum(counts) == np.count_nonzero(maskc), "Mismatch in selected core numbers"

        if np.count_nonzero(mask) > 0:
            msg = '{} <= ncores <= {}'.format(np.min(counts), np.max(counts))
            print('Selecting {} parents with {}'.format(np.count_nonzero(mask), msg))
            print('Time to find selected parents = {:.2g} minutes'.format((time() - htime)/60.))

            # loop over parent halos and process siblings together
            ptime1 = time()
            #numba attempts
            #coretrees = process_parent_halos(coretrees, parent_halos, index, counts, sorted_indices_this)
            #stimes.append(stime)
            #atimes.append(atime)
            
            for npar, (p, indx, cnt) in enumerate(zip(parent_halos, index, counts)):
                # get sibling tags, array locations and masses
                stime1 = time()
                siblings = coretags_fofm_this[indx:indx+cnt]  # values from sorted array
                locations = sorted_indices_this[indx:indx+cnt] # locations in corecat arrays
                
                stimes.append((time()-stime1)/60.)
                atime1 = time()
                for ns, (s, loc) in enumerate(zip(siblings, locations)):
                    next_sibling = siblings[ns+1] if ns < len(siblings)-1 else -1
                    if not coretrees or s not in list(coretrees.keys()): #first snapshot or first instance of s
                        coretrees[s] = {}
                    coretrees[s] = add_properties_to_tree(s, loc, coretrees[s], corecat, properties,
                                                          next_sibling, siblings[0],
                                                          snap_index, current_snap, p,
                                                          fmt=fmt, particle_mass=particle_masses[sim])
                atimes.append((time()-atime1)/60.)
                if npar % print_int == 0 and npar > 0:
                    print('Time to loop over {} parents = {:.2g} minutes'.format(npar, (time() - ptime1)/60.))
            
            
            print('Time to loop over selected parents = {:.2g} minutes'.format((time() - ptime1)/60.))

        # fix first progenitors if no core in this step
        coretags_not_this = get_coretags_not_this(selected_cores,  coretags_fofm_this.tolist())
        #coretags_not_this = get_coretags_not_this(np.asarray(selected_cores),  coretags_fofm_this)
        coretrees = fix_first_progenitors(coretrees, coretags_not_this, snap_index)
            
    return coretrees, atimes, stimes


#@numba.jit(nopython=True)
def get_coretags_not_this(selected_cores,  coretags_fofm_this):
    #coretags_not_this = [c for c in selected_cores if c not in coretags_fofm_this.tolist()]
    coretags_not_this = []
    for c in selected_cores: 
        if c not in coretags_fofm_this:
            coretags_not_this.append(c)
    
    return coretags_not_this

#@numba.jit(nopython=True)
def process_parent_halos(coretrees, parent_halos, index, counts, sorted_indices_this):

    for npar, (p, indx, cnt) in enumerate(zip(parent_halos, index, counts)):
        # get sibling tags, array locations and masses
        #stime1 = time()
        siblings = coretags_fofm_this[indx:indx+cnt]  # values from sorted array
        locations = sorted_indices_this[indx:indx+cnt] # locations in corecat arrays
                
        #stime = (time()-stime1)/60.
        #atime1 = time()
        for ns, (s, loc) in enumerate(zip(siblings, locations)):
            next_sibling = siblings[ns+1] if ns < len(siblings)-1 else -1
            if not coretrees or s not in list(coretrees.keys()): #first snapshot or first instance of s
                coretrees[s] = {}
            coretrees[s] = add_properties_to_tree(s, loc, coretrees[s], corecat,
                                                  next_sibling, siblings[0],
                                                  snap_index, current_snap, p)
            #atime = (time()-atime1)/60.
            #if npar % print_int == 0 and npar > 0:
                #print('Time to loop over {} parents = {:.2g} minutes'.format(npar, (time() - ptime1)/60.))

    return coretrees
                
def get_sibling_vectors(coretags, index, inverse):
    """
    coretags: array sorted in decreasing tag, mass order
    index, inverse, counts: output of np.unique
    """
    # find most massive core corresponding to parent halos
    first_sibling_coretags = coretags[index]
    # use inverse to reconstruct first-sibling array
    first_siblings = first_sibling_coretags[inverse]

    # initialize array and pointers for next siblings
    ncores = len(coretags)
    nparents = len(index)
    next_siblings = np.array([-1]*ncores) #null pointer
    last_sibling_loc = np.zeros(nparents - 1).astype(int)
    # shift coretag array by 1 argument to the left; last element is left as -1
    next_siblings[0:ncores-1] = coretags[1:ncores]
    # shift index locations by -1 to point to last sibling in each group
    last_sibling_loc[0:nparents-1] = index[1:nparents] - 1
    next_siblings[last_sibling_loc] = -1
    
    return first_siblings, next_siblings

# fix first progenitors
def fix_firstprogenitor_vector(first_progenitor, row):

    if row > 0:
        mask_this = (first_progenitor[row] == no_int) #no entry in this snap
        mask_prev = (first_progenitor[row-1] != no_int) #entry in previous snap
        mask = mask_this & mask_prev
        first_progenitor[row-1][mask] = -1 #overwrite
    return first_progenitor

def get_ordered_property(p, corecat, sorted_indices_this, row, current_snap, properties,
                         first_siblings, next_siblings, parent_tags,
                         fmt=core, particle_mass=particle_masses[default_sim]):
    ncores = len(corecat[coretag]) # = len(sorted_indices_this)
    if fmt == lgal:
        a = get_a(current_snap)

    # assign pointers
    if Descendent in p:
        prop_values = np.array([row - 1]*ncores) #row = row of matrix array
    elif FirstProgenitor in p:
        prop_values = np.array([row + 1]*ncores) if row != last_row else np.array([-1]*ncores)
    elif NextProgenitor in p:
        prop_values = np.array([-1]*ncores)
    elif FirstHaloInFOFGroup in p:
        prop_values = first_siblings
    elif NextHaloInFOFGroup in p:
        prop_values = next_siblings
    # assign derived properties
    elif SnapNum in p:
        #prop_values = np.array([current_snap]*ncores)
        prop_values = np.array([last_row - row]*ncores) #L-galaxies needs consecutive integers
    elif timestep in p:
        prop_values = np.array([current_snap]*ncores)
    elif ParentHaloTag in p:
        prop_values = parent_tags
    elif Len in p:
         prop_values = np.round(corecat[coremass][sorted_indices_this]/particle_mass).astype(int) #truncate
         mask = (prop_values == 0)
         prop_values[mask] = 1  #ensure non-zero values
    elif p in properties['int']:
        if Zero in properties['read'][p]:
            prop_values = np.zeros(ncores).astype(int)
        else:
            prop_values = corecat[properties['read'][p]][sorted_indices_this] # reorder into sorted coretag order
            #print('Prop values for {}: {} {} from corecat {} {}'.format(p, prop_values.dtype, prop_values[0],
            #                                                            corecat[properties['read'][p]].dtype,
            #                                                            corecat[properties['read'][p]][0]))
    elif p in properties['float']:
        if Zero in properties['read'][p]:  #zero values
            prop_values = np.zeros(ncores)
        else:
            prop_values = corecat[properties['read'][p]][sorted_indices_this] # reorder into sorted coretag order
            #fix normalizations
            if 'M_Crit' in p:
                mask = (corecat['central'][sorted_indices_this]==0) # select non-centrals
                prop_values[mask] = 0.   # set satellite masses to 0.
                prop_values[~mask] /= M_Crit_norm
            if 'Spin' in p:  #normalize by infall_sod_halo mass
                sod_mask = corecat[infall_sod_halo_mass][sorted_indices_this] > 0  #find valid mass entries
                prop_values[sod_mask] /= corecat[infall_sod_halo_mass][sorted_indices_this][sod_mask]
                prop_values[~sod_mask] = no_mass
            if fmt == lgal and p in list(a_scaling[fmt].keys()):
                prop_values *= a**(a_scaling[fmt][p])
                
    else:
        print('Unknown property {}'.format(p))

    return prop_values


# fix serial trees first progenitors
#@numba.jit(nopython=True)
def fix_first_progenitors(coretrees, coretags_not_this, snap_index, fmt=core):

    for s in coretags_not_this:  #loop thru trees without entry in this snapshot
        if len(coretrees[s][FirstProgenitor]) == snap_index: #but have entry in last snapshot
            coretrees[s][FirstProgenitor][-1] = -1 #overwrite last element in list
            if fmt == lgal:
                coretrees[s][FirstProgenitorOffset][-1] = -1
            
    return coretrees

#@numba.jit(nopython=True)
def add_properties_to_tree(core_tag, location, coretree, corecat, properties,
                           next_sibling, first_sibling, snap_index,
                           current_snap, parent_fof_tag,
                           fmt=core, particle_mass=particle_masses[default_sim]):
    if not coretree:  # empty - first entry
        coretree[Descendent] = [-1] # no descendent for first halo
        coretree[FirstProgenitor] = [1] # will be next element (if it exists)
        coretree[NextProgenitor] = [-1] # no Next_Progenitor since cores are unmerged
        if fmt == lgal:
            coretree[DescendentOffset] = [-1] # no descendent for first halo
            coretree[FirstProgenitorOffset] = [1] # will be next element (if it exists)
            coretree[NextProgenitorOffset] = [-1] # no Next_Progenitor since cores are unmerged
        
        # initialize empty lists for properties
        for p in sibling_pointers + list(properties['read'].keys()) + derived_properties:
            coretree[p] = []
    else:
        # descendent will be next core unless in snap 499
        array_len = len(coretree[Descendent])
        coretree[Descendent].append(array_len - 1)
        coretree[DescendentOffset].append(array_len - 1)
        # now fill in location of next element in tree
        if current_snap != last_snap:
            coretree[FirstProgenitor].append(array_len + 1)
            coretree[FirstProgenitorOffset].append(array_len + 1)
        else:
            coretree[FirstProgenitor].append(-1)
            coretree[FirstProgenitorOffset].append(-1)
        coretree[NextProgenitor].append(-1)
        coretree[NextProgenitorOffset].append(-1)
    
    # add supplied properties; locations in other trees not known yet
    coretree[FirstHaloInFOFGroup].append(first_sibling)
    coretree[NextHaloInFOFGroup].append(next_sibling)
    if fmt == lgal:
        coretree[FirstHaloInFOFGroupOffset].append(first_sibling)
        coretree[NextHaloInFOFGroupOffset].append(next_sibling)
    # other properties
    if SnapNum in properties['derived']:
        coretree[SnapNum].append(last_row - snap_index)
    if timestep in properties['derived']:
        coretree[timestep].append(current_snap)
    if ParentHaloTag in properties['derived']:  
        coretree[ParentHaloTag].append(parent_fof_tag)
    if Len in properties['derived']:
        coretree[Len].append(int(max(np.round(corecat[coremass][location]/particle_mass), 1.))) #set min mass to 1 particle
    
    for p, v in properties['read'].items():
        if Zero in v:
            if p in properties['int']:
                coretree[p].append(0)
            else:
                coretree[p].append(0.)
        else:
            coretree[p].append(corecat[v][location])
            #fix normalizations
            if 'M_Crit' in p:
                if corecat['central'][location] == 0:
                    coretree[p][-1] = 0.  #overwrite last entry with zero for satellite
                else:
                    coretree[p][-1] /= M_Crit_norm
            if 'Spin' in p:
                if corecat[infall_sod_mass] > 0:
                    coretree[p][-1] /= corecat[infall_sod_halo_mass][location] # divide by mass
                else:
                    coretree[p][-1] = no_mass
            if fmt == lgal and p in list(a_scaling[fmt].keys()):
                coretree[p][-1] *= a**(a_scaling[fmt][p])

    return coretree

def get_parent_boundary(foftags, loc, ncores, name='input', upper=True):
    # assumes foftags are in fofm order
    oldloc = loc
    foftag_this = foftags[loc-1] # parent of preceding index
    args =  np.where(foftags == foftag_this)[0] #locations of this parent
    if upper:
        loc = min(np.max(args) + 1, ncores) # maximum argument +1
    else:
        loc = min(np.min(args) - 1, 0)
    if loc != oldloc:
        print('Resetting {} to {} to line up with parent boundary'.format(name, loc))
    
    return loc


def overwrite_fragment_quantities(coretrees, Nsnaps, replace=['Vmax', 'Spin_x', 'Spin_y', 'Spin_z'], vector=True):
    if vector:
        for s in np.arange(0, Nsnaps)[::-1]: #start with earliest snapshot
            valid = (coretrees[MostBoundID_Coretag][s] != no_int)  #remove placeholder values
            maskf = (coretrees[fof_halo_tag][s][valid] < 0)  #find -ve halo tags 
            if np.count_nonzero(maskf) == 0:
                print('No fragments in {} halos in snapnum {}'.format(np.count_nonzero(valid), s))
                continue
            print('Processing {} fragments in {} halos in snapnum {}'.format(np.count_nonzero(maskf), np.count_nonzero(valid), s))
            if s == Nsnaps:
                print('  Fragments in earliest snapshot cannot be updated with earlier values: skipping')
                continue
            # loop over -ve halo tags, decode and find original halos and  associated cores
            original_fof_tags = byte_mask(coretrees[fof_halo_tag][s][valid][maskf])
            frag_tags = shift_tag(coretrees[fof_halo_tag][s][valid][maskf]) 
            fragment_masses = coretrees[Len][s][valid][maskf]
            # identify small fragments
            small_frag_mask = (frag_tags > 0)
            small_frag_halo_tags = original_fof_tags[small_frag_mask]
            small_frag_masses = fragment_masses[small_frag_mask]
            coretags = coretrees[MostBoundID_Coretag][s][valid][maskf][small_frag_mask]
            # replace values
            print('  Processing {} LMFs'.format(np.count_nonzero(small_frag_mask)))
            missing_mmf = 0
            for coretag, halo_tag, frag_mass in zip(coretags, small_frag_halo_tags, small_frag_masses):
                #check that it is a less massive fragment (less massive than fragment 0)
                mmf_mask = (original_fof_tags == halo_tag) & (frag_tags == 0)
                if np.count_nonzero(mmf_mask) > 0:
                    mmf_mass = fragment_masses[mmf_mask][0] #only element in array
                    mmf_coretag = coretrees[MostBoundID_Coretag][s][valid][maskf][mmf_mask][0]
                    if mmf_mass < frag_mass:
                        print('    Warning: MMF with halotag {} and coretag {}  has mass {} < fragment with coretag {} and mass {}'.format(halo_tag,
                                                                                                                                           mmf_coretag,
                                                                                                                                           mmf_mass,
                                                                                                                                           coretag,
                                                                                                                                           frag_mass))
                else:
                    missing_mmf +=1
                column = np.where(coretrees[MostBoundID_Coretag][s] == coretag)[0][0]  #locate column
                mask_col = (coretrees[fof_halo_tag][:,column][s+1:Nsnaps] > 0)         #find earlier positive halo_tag locations
                row = np.where(mask_col == True)[0][0]        #get first location
                #overwrite selected quantities with earlier non-fragment value
                for r in replace:
                    #print('Replacing {} for coretag {} in snap {}: {} -> {}'.format(r, coretag, s,
                    #                                                                coretrees[r][s][column],
                    #                                                                coretrees[r][s + 1 + row][column]))
                    coretrees[r][s][column] = coretrees[r][s + 1 + row][column]
                    
            if missing_mmf > 0:
                n_small_frag = np.count_nonzero(small_frag_mask)
                missing_frac = missing_mmf/np.count_nonzero(maskf)
                print('    Warning: Found {}/{} LMFs with no MMF counterpart in snap {}\n    Fraction of all fragments = {:.3g}'.format(missing_mmf,
                                                                                                                                       np.count_nonzero(small_frag_mask),
                                                                                                                                       s, missing_frac))
    else:    
        print('Fragment overwite not implemented yet\n')

    return coretrees


def byte_mask(tag_val, bit_mask=0xffffffffffff): 
    if (tag_val<0).all() == False:
        print("Warning: fof tag passed to byte masking routine, this assumes fragment tags only")
    return  (-tag_val)&bit_mask


def shift_tag(tag_val, shift=48):
    if (tag_val<0).all() == False:
        print("Warning: fof tag passed to byte masking routine, this assumes fragment tags only")
    return  np.right_shift(-tag_val, 48)


def write_outfile(outfile, coretrees, cores_to_write, vector=False, start=None, end=None,
                  column_counts=None):
    """
    coretrees: trees to output
    cores_to_write: list of sorted cores
    vector: True if using vectorized code (requires start and end arguments)
    start: starting column of coretree matrix
    end: ending column of coretree matrix (= parent_boundary+1) 
    """
    wtime = time()
    if os.path.exists(outfile):
        os.remove(outfile)
        print("Removed old file {}".format(outfile))
    if len(cores_to_write) == 0:
        return
    hdfFile = h5py.File(outfile, 'w')
    hdfFile.create_group('treeData')
    hdfFile['treeData']['Ntrees'] = len(cores_to_write)
    # use Descendent to get lengths, masks etc. 
    if vector:
        if column_counts is None:
            print('Need column counts')
            return
        hdfFile['treeData']['TreeNHalos'] = column_counts
    else:
        hdfFile['treeData']['TreeNHalos'] = np.asarray([len(coretrees[k][Descendent]) for k in cores_to_write]) 
    hdfFile['treeData']['coretags'] = cores_to_write
    hdfFile['treeData']['totNHalos'] = np.sum(hdfFile['treeData']['TreeNHalos'])
    
    print('Number of core trees = {}, Total halos = {}'.format(hdfFile['treeData']['Ntrees'][()],
                                                          hdfFile['treeData']['totNHalos'][()]))
    
    # concatenate trees
    tgroup = hdfFile.create_group('trees')
    if vector:
        mtime = time()
        properties_list = list(coretrees.keys())
        mask = (coretrees[Descendent][:, start:end] != -999)  # mask for non-existent elements
        flat_mask = mask.flatten(order='F')
        print('Time to build flat mask = {:.2e} minutes'.format((time() - mtime)/60.))
    else:
        properties_list = list(coretrees[cores_to_write[0]].keys())

    ptime = time()
    for p in properties_list:
        if vector:
            prop_array = coretrees[p][:, start:end].flatten(order='F') 
            tgroup[p] = prop_array[flat_mask]
        else:
            tgroup[p] = concatenate_trees(coretrees, cores_to_write, p)
            
        assert len(tgroup[p][()]) == hdfFile['treeData']['totNHalos'][()], \
            'Length mismatch for {}'.format(p,  len(tgroup[p][()]), hdfFile['treeData']['totNHalos'][()])
        print('Assembled group {} in {:.2e} minutes'.format(p, (time() - ptime)/60.))

    print('Time to write trees to {} = {:.2e} minutes'.format(outfile, (time() - wtime)/60.))
    hdfFile.close()
                                      
def get_column_counts(coretree_matrix_transpose):
    #vectorize
    counts = np.zeros(len(coretree_matrix_transpose)).astype(int)
    for n, c in enumerate(coretree_matrix_transpose):
        mask = (c != -999)
        counts[n] = np.count_nonzero(mask)

    return counts
                                          
def concatenate_trees(coretrees, keys, p):
    if p in integer_properties:
        prop_array = np.asarray([]).astype(int)
    else:
        prop_array = np.asarray([])

    for k in keys:
        v = np.asarray(coretrees[k][p])
        prop_array = np.concatenate((prop_array, v))
        
    return prop_array

def read_outfile(outfile, vector=True):
    with h5py.File(outfile, 'r') as fh:
        Ntrees = fh['treeData']['Ntrees'][()]
        totNHalos = fh['treeData']['totNHalos'][()]
        TreeNHalos = fh['treeData']['TreeNHalos'][()]
        coretags = fh['treeData']['coretags'][()]
        assert len(TreeNHalos) == Ntrees, 'Mismatch in number of trees, {} != {}'.format(len(TreeNHalos), Ntrees)

        coretrees = {}
        if not vector:
            offset=0
            properties = list(fh['trees'].keys())
            for core, nhalos in zip(coretags, TreeNHalos):
                coretrees[core] = {}
                for p in properties:
                    coretrees[core][p] = fh['trees'][p][offset:offset + nhalos] 

                offset = offset + nhalos
        else:
            for k in fh['trees'].keys():
                coretrees[k] = fh['trees'][k][()]
                
    return coretrees, Ntrees, totNHalos, TreeNHalos, coretags
        
def read_binary(outfile, vector=True):
    nbytes = (len(struct_keys)-1)*4 + 1*8  # account for 1 long word in struct
    dtype = [(struct_keys[k], struct_format[k+1]) for k in range(len(struct_keys))]
    trees = {}
    with open(outfile, 'rb') as fh:
        # read header
        Ntrees = struct.unpack('<i', fh.read(4))[0]
        totNhalos = struct.unpack('<i', fh.read(4))[0]
        halos_per_tree = np.asarray(struct.unpack('<{}i'.format(Ntrees), fh.read(4*Ntrees)))
        for tree, nhalos in zip(np.arange(Ntrees), halos_per_tree):
            #initialize
            data_list = []
            # read tree data
            for n in np.arange(nhalos):
                values = struct.unpack(struct_format, fh.read(nbytes))
                data_list.append(values)

            trees[tree] = np.array(data_list, dtype=dtype)
            #print('Read tree {} with {} halos'.format(tree, nhalos))
                
        print('Read {} trees with total {} halos'.format(Ntrees, totNhalos))

    return trees, Ntrees, totNhalos, halos_per_tree

def write_binary(outfile, coretrees, cores_to_write, foftags_to_write,
                 vector=True, start=None, end=None, column_counts=None):
    """
    coretrees: trees to output
    cores_to_write: list of sorted cores
    vector: True if using vectorized code (requires start and end arguments)
    start: starting column of coretree matrix
    end: ending column of coretree matrix (= parent_boundary+1) 
    column_counts: counts of cores in each tree to be written
    """

    wtime = time()
    rtimes = []
    if column_counts is None:
        print('Need column counts')
        return
    if len(cores_to_write) == 0:
        print('No cores to write')
        return
        
    with open(outfile, "wb") as fh:
        # L-galaxies requires trees organized by parent groups
        fof_groups, index, fof_counts = np.unique(foftags_to_write, return_index=True, return_counts=True)
        fof_group_order = index.argsort()   #reorder parent halos back to original order
        #get number of halos in each fof_group
        halos_per_tree = get_halo_counts(fof_counts[fof_group_order], column_counts)
        
        # write global information for file
        Ntrees = len(fof_groups)
        totNHalos = int(np.sum(column_counts))        
        values = [Ntrees, totNHalos] + halos_per_tree.tolist()
        header_this = header_format.format(Ntrees + 2)
        fh.write(pack(header_this, *values))
        print('Wrote header for {} FoF trees with {} halos'.format(Ntrees, totNHalos))
        
        if vector:
            assert np.array_equal(foftags_to_write, coretrees[ParentHaloTag][0, start:end]), "ParentHaloTags not in fofm order"
            validate_trees(column_counts, coretrees, start, end)
            #TODO validate parent tags and FirstHaloTags (order will be same but tags will not)
            
            # write struct for each tree (ie forest) (cores in same parent halo at z=0)
            column = start
            for fof_count in fof_counts[fof_group_order]:
                offset = 0   #reset offset for each fof group
                for col, count in zip(np.arange(column, column + fof_count),
                                      column_counts[column-start:column-start+fof_count]):  #column counts are restarted for each file
                    coretrees = add_offsets(coretrees, col, offset, count)
                    rtime = time()
                    coretrees = replace_sibling_addresses(coretrees, col, column_counts,
                                                          start, column)
                    rtimes.append(time() - rtime)
                    for row in np.arange(count):
                        values = [coretrees[p].T[col][row] for p in struct_keys]
                        fh.write(pack(struct_format, *values))
                    offset = offset + count   #offset addresses by # of rows in column
                    
                column += fof_count    

        else:
            #validate that ParentHaloTags match FOF group
            parent_halo_tags = np.asarray([coretrees[s][ParentHaloTag] for s in cores_to_write])
            #assert np.array_equal(parent_halo_tags, foftags_to_write) 
            print(parent_halo_tags, foftags_to_write)
            
            corecount = start
            for fof_group, fof_count in zip(fof_group[fof_group_order], fof_counts[fof_group_order]):
                offset = 0
                cores_in_group = cores_to_write[corecount:corecount + fof_count]
                group_counts = column_counts[corecount:corecount + fof_count]
                for core, count in zip(cores_in_group, group_counts):
                    coretrees[core] = add_offsets_serial(coretrees[core], offset)
                    rtime = time()
                    coretrees[core] =  replace_sibling_addresses_serial(core, count, coretrees,
                                                                        cores_in_group, group_counts)
                    rtimes.append(time() - rtime)
                    for snap in np.arange(count):
                        values = [coretrees[core][p][snap] for p in struct_keys]
                        fh.write(pack(struct_format, *values))
                    offset = offset + count #offset addresses by

                corecount += fof_count
                
    print('Time to compute sibling addresses = {:.2e} minutes'.format(sum(rtimes)/60.))
    print('Time to write trees to {} = {:.2e} minutes'.format(outfile, (time() - wtime)/60.))
  
    return

def get_halo_counts(counts, column_counts):
    # sum up halos in each fof group
    
    halos_per_tree = np.zeros(len(counts)).astype(int)
    offset = 0
    for n, count in enumerate(counts):
        halos_per_tree[n] = np.sum(column_counts[offset:offset+count])
        offset += count

    return halos_per_tree

def validate_trees(counts, coretrees, start, end):
    
    # check coretag is unique and equal to counts
    for count, col in zip(counts, np.arange(start, end)):
        unique_tags = np.unique(coretrees[MostBoundID_Coretag].T[col][0:count])
        assert len(unique_tags)==1, 'Mutiple core tags {} found in column {}'.format(unique_tags,
                                                                                     col)

    print('Validated coretrees[{}:{}] to have unique coretags'.format(start, end))
    return

def add_offsets(coretrees, col, offset, count):
    #change value in "Offset" arrays
    #add offset to coretree column if element != -1
    for p in [Descendent, FirstProgenitor]: #, NextProgenitor]:
        mask = coretrees[p][:, col] > -1
        coretrees[p + Offset][:, col][mask] += offset
        
    return coretrees

def add_offsets_serial(coretree, offset):
    #change value in "Offset" arrays
    #add offset to coretree if element != -1
    for p in [Descendent, FirstProgenitor]: #, NextProgenitor]:
        pvals = np.asarray(coretree[p])
        mask = (pvals > -1)
        pvals[mask] += offset
        coretree[p + Offset] = pvals

    return coretree

def replace_sibling_addresses(coretrees, col, column_counts, start, first_column_in_tree):
    #change value in "Offset" arrays
    for p in [FirstHaloInFOFGroup, NextHaloInFOFGroup]:
        cindx = col - start # correct for offset in column counts (evaluated for each file)
        mask = (coretrees[p][:, col][0:column_counts[cindx]] > 0)
        #print(col, column_counts[cindx], coretrees[p][:, col])
        if np.count_nonzero(mask) > 0:  # check if any entries are coretags
            unique_sibs, inverse = np.unique(coretrees[p][:, col][0: column_counts[cindx]], return_inverse=True)
            rows = np.arange(0, column_counts[cindx]) # vector of row indices
            #print(col, unique_sibs, inverse, rows)
            for ns, sib in enumerate(unique_sibs):
                if sib > 0:   #skip -1 entries
                    loc = np.where(coretrees[MostBoundID_Coretag][0]==sib)[0][0]
                    if loc < first_column_in_tree:
                        print("For p {}: {}th sibling {} with location {} outside of {}th tree ".format(p, ns, sib, loc, col))
                        print("Substituting pointer to first halo in tree")
                        loc = first_column_in_tree
                    offset = np.sum(column_counts[first_column_in_tree:loc])
                    sib_mask = (inverse==ns) & mask
                    coretrees[p + Offset][:, col][0:column_counts[cindx]][sib_mask] = offset + rows[sib_mask]
                
    return coretrees

def replace_sibling_addresses_serial(core, count, coretrees, cores_to_write, counts):
    #change value in "Offset" arrays
    for p in [FirstHaloInFOFGroup, NextHaloInFOFGroup]:
        siblings = np.asarray(coretrees[core][p])
        mask = (siblings > 0)
        if np.count_nonzero(mask) > 0:  # check if any entries are coretags
            unique_sibs, inverse = np.unique(siblings, return_inverse=True)
            addresses = np.arange(0, count)
            for ns, sib in enumerate(unique_sibs):
                if sib > 0:   #skip -1 entries
                    loc = np.where(cores_to_write==sib)[0][0]
                    offset = np.sum(counts[0:loc])
                    sib_mask = (inverse==ns)
                    addresses[sib_mask] = offset + addresses[sib_mask]

            coretrees[core][p + Offset] = addresses.tolist()
        
    return coretrees[core]

def replace_sibling_addresses_old(coretrees, col, column_counts):
    #change value in "Offset" arrays
    for p in [FirstHaloInFOFGroup, NextHaloInFOFGroup]:
        for row in np.arange(0, column_counts[col]):
            sibling = coretrees[p][row][col]
            if sibling > 0:
                loc = np.where(coretrees[MostBoundID_Coretag][0]==sibling)[0][0]
                offset = np.sum(column_counts[0:loc])
                # replace with address
                coretrees[p + Offset][row][col] = offset + row
    
    return coretrees


def main(argv):
    #setup parameters
    if type(argv) == list:
        argv = dict(zip(np.arange(0, len(argv)), argv))
    print('Inputs:',argv)
    name = argv.get(0, 'test')    
    vector = True if argv.get(1, 'vector')=='vector' else False
    nfiles = int(argv.get(2, 3))   #number of files to write
    fmt = argv.get(3, core) # set format
    Nfiles = int(argv.get(4, 1000)) #total number of files for vector code
    print_int = int(argv.get(5, 1000)) #for serial code
    ncore_min = int(argv.get(6, 0))    #for serial code/HM code
    ncore_max = int(argv.get(7, 67760)) #for serial code/HM code (number of cores)/100 #33005401 in HM (~5x LV-SJ)
    sim = argv.get(8, 'SV')  # set simulation

    coredir = '../CoreCatalogs_{}'.format(sim)
    outname = '{}_{}'.format(sim, name) if 'test' in name else sim 
    treedir = '../CoreTrees/fof_group_{}_{}'.format(outname, fmt)
    #setup output file template
    outfile_template = re.sub('properties', 'trees', cc_template[sim])
    outfile_template = re.sub('extend', '', outfile_template)
    outfile_template = outfile_template + '.hdf5' if 'hdf5' not in outfile_template else outfile_template
    
    print('Simulation = {}'.format(sim))
    #evaluate exact particle mass
    particle_masses[sim] = get_particle_mass(sim)
    print('Outputs written to {}'.format(treedir))
    print('Writing {} file(s) out of a total of {}'.format(nfiles, Nfiles))
    print('Reading core catalog from {}'.format(coredir))
    
    process = psutil.Process(os.getpid())
    
    corefiles = glob.glob(coredir+'/*')
    snapshots = sorted([int(os.path.basename(f).split('.')[0].split('-')[-1]) for f in corefiles], reverse=True)
    coretrees = {}

    #get list of properties depending on format selected
    properties = assemble_properties(fmt, vector)

    for n, s in enumerate(snapshots): #process in descending order
        print('Processing snapshot {}'.format(s))
        stime = time() 
        corecat = get_core_snapshot(coredir, int(s), sim=sim, template=cc_template[sim])
        if corecat:
            if sim != 'HM':
                corecat = add_mass_columns(corecat, sim=sim)
            if n == 0:
                sorted_coretags = np.sort(corecat[coretag]) #save for cleaning earlier snaps

                # sort corecat first by foftag and then coremass so siblings are grouped together
                indices_fofm = np.flip(np.lexsort(((corecat[coremass], corecat[foftag]))))
                coretags_fofm = corecat[coretag][indices_fofm] #order of coretags in coretrees
                foftags_fofm = corecat[foftag][indices_fofm]
                Ncores = len(coretags_fofm)
                argsorted_coretags_fofm = coretags_fofm.argsort()
                if vector: # initialize dict of matrices
                    print('Setting up ordered arrays and tree matrices')
                    ctime = time()
                    for p in properties['int']:
                        if p in properties_int32:
                            coretrees[p] = np.array([no_int]*Ncores*len(snapshots)).reshape(len(snapshots), Ncores).astype(np.int32)
                        else:
                            coretrees[p] = np.array([no_int]*Ncores*len(snapshots)).reshape(len(snapshots), Ncores).astype(np.int64)
                        print('Initial {} value for {}: {}'.format(coretrees[p].dtype, p, coretrees[p][0][0]))
                    for p in properties['float']:
                        coretrees[p] = np.array([no_float]*Ncores*len(snapshots)).reshape(len(snapshots), Ncores).astype(np.float64)
                        #print('Initial value for {}: {:.3g}'.format(p, coretrees[p][0][0]))
                        
                    print('Time to create arrays = {:.2f} minutes'.format((time() - ctime)/60.))
                    print('Keys are: {}'.format(sorted(list(coretrees.keys()))))
                else:
                    # check ncore_max lands on parent boundary; need to check ncore_min if != 0
                    ncore_max = get_parent_boundary(foftags_fofm, ncore_max, Ncores, name='ncore_max')
            else:
                # clean corecat to remove cores that disappeared before step 499
                corecat = clean(corecat, sorted_coretags)
            
            coretrees, atimes, stimes = add_snapshot_to_trees(coretrees, corecat, properties, int(s), sim,
                                                              snap_index=n, coretags_fofm=coretags_fofm,
                                                              argsorted_fofm=argsorted_coretags_fofm,
                                                              sorted_indices=indices_fofm, fmt=fmt,
                                                              print_int=print_int, ncore_max=ncore_max,
                                                              ncore_min=ncore_min, vector=vector)

            if len(atimes) > 0:
                print('Min/max times to run siblings = {:.3g}/{:.3g}'.format(min(stimes), max(stimes)))
                print('Min/max times to run add_props = {:.3g}/{:.3g}'.format(min(atimes), max(atimes)))
                amean = np.mean(np.asarray(atimes))
                smean = np.mean(np.asarray(stimes))
                if vector:
                    print('Mean times = {:.3g}(sib); {:.3g}(add)'.format(smean, amean))
                else:
                    print('Mean times (over {} parents) = {:.3g}(sib); {:.3g}(add)'.format(len(atimes), smean, amean))

        print('Time to run snapshot = {:.2f} minutes'.format((time() - stime)/60.))
        mem = "Memory usage =  {0:.2f} GB"
        print(mem.format(process.memory_info().rss/1.e9))
        
    del corecat

    # deal with fragments
    # must be done after the first pass through the trees becase we need to use information from earlier timesteps
    coretrees = overwrite_fragment_quantities(coretrees, len(snapshots), vector=vector)

    # TODO merging
    # TODO truncate for halo trees
    
    # output core trees
    numcores = Ncores if vector else len(list(coretrees.keys()))
    numfiles = Nfiles if vector else nfiles
    stride = int(np.ceil(numcores/numfiles))
    start = 0 if vector else ncore_min
    mode = '.vector' if vector else '.serial'
    print('Writing subsets of {} cores in {} files with stride {}'.format(numcores, nfiles, stride))
    
    for n in range(nfiles):
        fn = os.path.join(treedir, outfile_template.format('', n)+mode)
        fn_bin = os.path.join(treedir, binfile_template.format(n)+mode)
        end = int(min(start + stride, numcores))
        # check that end is on a parent boundary
        end = get_parent_boundary(foftags_fofm, end, Ncores, name='end')
          
        cores_to_write = coretags_fofm[start:end]
        foftags_to_write = foftags_fofm[start:end]
        if len(cores_to_write) > 0:
            print('n/start/end/#cores: {} {} {} {}'.format(n, start, end, len(cores_to_write)))
            if vector:
                column_counts = get_column_counts(coretrees[Descendent].T[start:end]) #get counts for unfilled elements
            else:
                column_counts = [len(coretrees[core][Descendent]) for core in cores_to_write]
            # write binary & hdf5
            write_outfile(fn, coretrees, cores_to_write, vector=vector, start=start, end=end,
                          column_counts=column_counts)
            if fmt == lgal:
                write_binary(fn_bin, coretrees, cores_to_write, foftags_to_write,
                             vector=vector, start=start, end=end, column_counts=column_counts)
        start = end
    
    # eg to run: serialtrees = build_core_trees.main({0:'test', 1:'serial', 2:2, 3:'lgal', 4:1000, 5:500, 6:0, 7:500})
    # binary write only for lgal format
    # coretrees = build_core_trees.main({0:'test_core', 1:'vector', 2:2})
    # coretrees = build_core_trees.main({0:'test_core', 1:'vector', 2:1, 3:'core': 4:20, 5:500, 6:0, 7:500, 8:'SV'})  #write out 1/20 files
    # testtree = build_core_trees.main({0:'test_lgal', 1:'serial', 2:1, 3:'lgal', 4:1000, 5:400, 6:246, 7:247})
    # eg to run from command line
    # python build_core_trees.py vector 2 lgal 20 500 0 500 SV |& tee ../logfiles/SV_v0.2_lgal.log
    # python build_core_trees.py vector 1 core 1 500 0 500 SV |& tee ../logfiles/SV_v0.2_core_all_11_06.log #write 1 file
    if 'return' in name:
        return coretrees
    else:
        return


color_cl = 'lightgrey'
outlinecolor='blue'
first_halo_arrow = 'seagreen'
next_halo_arrow = 'limegreen'
first_prog_arrow = 'orangered'
next_prog_arrow = 'cyan'
desc_arrow = 'orchid'
next_fof_arrow = 'blue'

# outfile='../CoreTrees/trees_099.0.vector'
# trees, Ntrees, totNhalos, halos_per_tree = build_core_trees.read_binary(outfile)
# coretrees, Ntrees, totNHalos, TreeNHalos, coretags = build_core_trees.read_outfile(outfile)
def get_mass_limits(trees, key=Len):
    mass_min = min([np.min(trees[i][key][trees[i][key] > 0.]) for i in range(len(trees)) if np.count_nonzero(trees[i][key] > 0.) > 0])
    mass_max = max([np.max(trees[i][key][trees[i][key] > 0.]) for i in range(len(trees)) if np.count_nonzero(trees[i][key] > 0.) > 0])
    mass_id = 'Core/Subhalo' if key==Len else 'Parent Halo'
    print('{} mass/m_p limits: min = {:.1f}; max = {:.1f}'.format(mass_id, mass_min, mass_max))
    return mass_min, mass_max

#Examples of usage
# treegraph, clusters, nodes = build_core_trees.drawforest(trees, 4)
# Note options mass_min, compresses, MTtrees
#treegraph, clusters, nodes, node_names, first_progenitors, next_progenitors = build_core_trees.drawforest(mttrees, 28727, compressed=False, xname='mt', MTtrees=True)
#treegraph, clusters, nodes, node_names, first_progenitors, next_progenitors = build_core_trees.drawforest(trees, 79, compressed=False, xname='reduced')
#treegraph, clusters, nodes, node_names, first_progenitors, next_progenitors = build_core_trees.drawforest(oldtrees, 79, compressed=False)

def drawforest(trees, treenum, filetype='.png', sim=default_sim, cmap='Purples',
               clmap='Wistia', clfont='black', fof_mass_min=None, fof_mass_max=None, fontcolor='black',
               mass_min=1, mass_max=None, alpha_halo=0.75, alpha_fof=0.5, compressed=True, MTtrees=False, xname=''): #pink_r is too light

    particle_mass=get_particle_mass(sim)
    tree = trees[treenum]
    if mass_max is None or mass_min is None:
        cmass_min, cmass_max = get_mass_limits(trees)
    mass_min = cmass_min if mass_min is None else mass_min
    mass_max = cmass_max if mass_max is None else mass_max
    if fof_mass_max is None or fof_mass_min is None:
        fmass_min, fmass_max = get_mass_limits(trees, key=M_Crit200)
    fof_mass_min = fmass_min if fof_mass_min is None else fof_mass_min
    fof_mass_max = fmass_max if fof_mass_max is None else fof_mass_max
        
    treegraph = pydot.Dot(graph_type="digraph", compound='true', rankdir="BT",
                          labelloc="c", labeljust="c", style="filled") #draws bottom to top only if newrank=True (see below)
    clusters = []
    nodes = []
    node_names= []
    logm_min = np.min(np.log10(tree[Len]*particle_mass))
    logm_max = np.max(np.log10(tree[Len]*particle_mass))
    print('Tree {}: log(m_min) = {:.2g}; log(m_max) = {:.2g}'.format(treenum, logm_min, logm_max))
    nsnaps = len(np.unique(tree['SnapNum']))
    norm = colors.LogNorm(vmin=mass_min*particle_mass, vmax=mass_max*particle_mass)
    cm = plt.get_cmap(cmap)
    norm_halo = colors.LogNorm(vmin=fof_mass_min*particle_mass, vmax=fof_mass_max*particle_mass)
    #get fof halo color map
    cm_halo = plt.get_cmap(clmap)
    # get code for rgb transparency
    trans_fof = hex(int(round(255*alpha_fof)))[-2:]
    trans_halo = hex(int(round(255*alpha_halo)))[-2:]
    
    first_progenitors = []
    next_progenitors = []
    snaps = np.arange(np.min(tree[SnapNum]), np.max(tree[SnapNum]+1))[::-1]

    #loop through snapshots; assign clusters and nodes
    for sidx, snap in enumerate(snaps):
        print('Processing snapshot {}'.format(snap))
        locs = np.where(tree[SnapNum]==snap)[0]
        #print('locs ', locs)
        coretags = tree[MostBoundID_Coretag][locs]
        masses = tree[Len][locs]*particle_mass
        lengths = tree[Len][locs]
        first_halos = tree[FirstHaloInFOFGroupOffset][locs]
        next_halos = tree[NextHaloInFOFGroupOffset][locs]
        unique_halos, index, counts = np.unique(first_halos, return_index=True,
                                                return_counts=True)
        orig_order = index.argsort()
        fof_groups = unique_halos[orig_order]
        halo_mlengths = tree[M_Crit200][fof_groups]
        halo_masses = tree[M_Crit200][fof_groups]*particle_mass
        labels_cl = ['FoF #{}; Coretag = {}'.format(str(f), tree[MostBoundID_Coretag][f]) for f in fof_groups]
        newrank = [f==0 for f in range(len(fof_groups))]
        if not MTtrees:  #add infall halo mass of fof group to label
            labels_cl = [labels_cl[i] + '\n M = {:.2e} M./h ({:.2f})'.format(halo_masses[i],
                                                                        halo_mlengths[i]) for i in range(len(fof_groups))]
            colors_rgb = [cm_halo(norm_halo(halomass)) for halomass in halo_masses]
            colors_cl = [colors.to_hex(c) + trans_fof for c in colors_rgb]
        else:
            colors_cl = [color_cl for f in fof_groups]

        # define subgraphs of clusters
        cl_this = [pydot.Cluster('FoF'+str(fof_groups[f]), label=labels_cl[f],
                                 color=colors_cl[f], fontcolor=clfont,
                                 newrank=newrank[f], rankdir='LR', rank='max') for f in range(len(fof_groups))] 

        #assign nodes to clusters and add edges
        nodes_this = []
        node_names_this = []
        first_progenitors_this = []
        next_progenitors_this = []
        for ncl, (cl, fof_group) in enumerate(zip(cl_this, fof_groups)):

            nodes_fof = []
            mask = (first_halos==fof_group)
            locs_in_fof = locs[mask]  #find locs in this fof_group
            coretags_in_fof = coretags[mask]
            masses_in_fof = masses[mask]
            lengths_in_fof = lengths[mask]
            first_progs = tree[FirstProgenitorOffset][locs][mask]
            next_progs = tree[NextProgenitorOffset][locs][mask]
            #add node
            for loc, ctag, mass, l, fpr, npr in zip(locs_in_fof, coretags_in_fof, masses_in_fof,
                                                  lengths_in_fof, first_progs, next_progs):
                node_name = str(loc) #label node by location
                node_label = "Node {}, Snap {}\n id={}\n M={:.2e} M./h (Len = {})".format(node_name, snap, ctag, mass, l)
                color_fp = colors.to_hex(cm(norm(mass))) + trans_halo
                node = pydot.Node(node_name, label=node_label, fillcolor=color_fp, style="filled",
                                  color=outlinecolor, fontcolor=fontcolor)
                cl.add_node(node)
                nodes_fof.append(node)
                # save nodes and node names and progenitors in identical order for ease in later searches
                nodes_this.append(node)
                node_names_this.append(node_name)
                first_progenitors_this.append(fpr)

                next_progenitors_this.append(npr)

            treegraph.add_subgraph(cl)  # add subgraphs for each cluster
            
            #now add edges for first and next halos and next_progenitors in this fof_group
            for nidx, (node, next_node_loc, next_prog_loc) in enumerate(zip(nodes_fof, next_halos[mask], next_progs)):
                # To save space, only add first halo arrow for first halo to itself if compressing plot
                if nidx==0 or not compressed:
                    treegraph.add_edge(pydot.Edge(node, nodes_fof[0], color=first_halo_arrow, style="dashed"))
                if next_node_loc != -1:
                    if next_node_loc in locs_in_fof:
                        nn_idx = locs_in_fof.tolist().index(next_node_loc)
                        treegraph.add_edge(pydot.Edge(node, nodes_fof[nn_idx], color=next_halo_arrow))
                    else:
                        print('Error: next halo {} for node {} not found in snap {}'.format(next_node_loc, node.get_name(), snap))

                #add edges for next progenitors in current snap (should account for same fof group instances in SV trees)
                if next_prog_loc != -1:
                    if next_prog_loc in locs_in_fof:   #is next progenitor in current snap in this fof_group?
                        np_idx = locs_in_fof.tolist().index(next_prog_loc)  #get its location
                        treegraph.add_edge(pydot.Edge(node, nodes_fof[np_idx], color=next_prog_arrow))  #arrow from node to location
                        #replace next prog with -1 since it has been accounted for
                        next_progenitors_this[nidx] = -1
                            
            #add invisible edge between last node in previous fof_group and first node in this group
            if ncl > 0:
                treegraph.add_edge(pydot.Edge(nodes_this[-len(nodes_fof) - 1], nodes_fof[0],
                                              ltail=cl_this[ncl-1].get_name(), lhead=cl.get_name(),  color=next_fof_arrow))
                                  
            #add edges for descendents progenitors and next progenitors connecting to nodes in this fof group
            if sidx > 0:
                #names = [nd.get_name() for nd in nodes_fof]
                #allsgraphs = treegraph.get_subgraph_list()
                #for sg in allsgraphs:
                #    print("Cluster",sg.get_name(),':',sg.get_attributes())
                #alledges = treegraph.get_edge_list()
                #for edge in alledges:
                #    print("Edge",edge.get_source(),"to",edge.get_destination(),edge.get_attributes())

                descendents = list(map(str, tree[DescendentOffset][locs_in_fof])) #all descendents in fof_group
                for node, desc in zip(nodes_fof, descendents):
                    # find desc node in any previous snapshots (for MT trees descendents may skip snapshots)
                    if MTtrees:
                        desc_loc = [(n, node_names[n].index(desc)) for n in range(len(node_names)) if desc in node_names[n]]
                    else:  #in previous snapshot
                        desc_loc = [(n, node_names[n].index(desc)) for n in range(sidx-1, sidx) if desc in node_names[n]]
                    if desc_loc:
                        if len(desc_loc) > 1:
                            print('Error: multiple descendant locations found for descendant {} for node {} in snap {}'.format(desc,
                                                                                                                               node.get_name(),
                                                                                                                               snap))
                        else:
                            treegraph.add_edge(pydot.Edge(node, nodes[desc_loc[0][0]][desc_loc[0][1]], color=desc_arrow))
                    else:
                        print('Error: decscendent node {} not found in snapshots < {} for node {}'.format(desc, snap, node.get_name()))
                                           
                    #search for node in progenitors or nextprogenitors by searching over previous snapshots
                    name = int(node.get_name())
                    #search progenitors for node name (= location)
                    if MTtrees:
                        fprog_loc = [(n, first_progenitors[n].index(name)) for n in range(len(first_progenitors)) if name in first_progenitors[n]]
                        nprog_loc = [(n, next_progenitors[n].index(name)) for n in range(len(next_progenitors)) if name in next_progenitors[n]]
                    else:  #search in previous time step
                        fprog_loc = [(n, first_progenitors[n].index(name)) for n in range(sidx-1, sidx) if name in first_progenitors[n]]
                        nprog_loc = [(n, next_progenitors[n].index(name)) for n in range(sidx-1, sidx) if name in next_progenitors[n]]
                        if nprog_loc:
                            print('Error: next progenitor {} for node {} found in previous snap {}'.format(nprog_loc, name,
                                                                                                           snaps[nprog_loc[0][0]]))
                    if fprog_loc:  #location in progenitor list matches location in nodes list
                        treegraph.add_edge(pydot.Edge(nodes[fprog_loc[0][0]][fprog_loc[0][1]], node, color=first_prog_arrow))
                        #replace this progenitor with -1 for consistency check at end of code
                        first_progenitors[fprog_loc[0][0]][fprog_loc[0][1]] = -1
                    if nprog_loc:
                        #add edge from previous node to progenitor 
                        treegraph.add_edge(pydot.Edge(nodes[nprog_loc[0][0]][nprog_loc[0][1]], node, color=next_prog_arrow))
                        #replace this progenitor with -1 for consistency check at end of code
                        next_progenitors[nprog_loc[0][0]][nprog_loc[0][1]] = -1

        if not all([n==-1 for n in next_progenitors_this]):  #next progenitors in same snapshot but in different fof group
            next_progs = [nxt for nxt in next_progenitors_this if str(nxt) in node_names_this]
            next_prog_idxs = [next_progenitors_this.index(nxt) for nxt in next_progenitors_this if str(nxt) in node_names_this]
            for next_prog, next_prog_idx in zip(next_progs, next_prog_idxs):
                next_node = nodes_this[node_names_this.index(str(next_prog))]
                treegraph.add_edge(pydot.Edge(nodes_this[next_prog_idx], next_node, color=next_prog_arrow))
                next_progenitors_this[next_prog_idx] = -1 #reset to -1

        nodes.append(nodes_this)
        node_names.append(node_names_this)
        clusters.append(cl_this)
        first_progenitors.append(first_progenitors_this)
        next_progenitors.append(next_progenitors_this)

        
    #check that progenitors are accounted for
    fp_check = [fp==-1 for sublist in first_progenitors for fp in sublist]
    np_check = [fp==-1 for sublist in next_progenitors for fp in sublist]
    if not all(fp_check):
        print('Error: not all first progenitors found')
    if not all(np_check):
        print('Error: not all next progenitors found')
        
    #write graph
    #dot_string = treegraph.to_string()

    if '.png' in filetype:
        treeid = '{}_{}'.format(treenum, xname) if len(xname) > 0 else treenum
        fn = '../pngfiles/tree_{}.png'.format(treeid)
        #fd = '../pngfiles/tree_{}.dot'.format(treeid)
        treegraph.write_png(fn)
        #treegraph.write_dot(fd)
        print('Wrote {}'.format(fn))

    return treegraph, clusters, nodes, node_names, first_progenitors, next_progenitors


def compare_trees(serialtrees, coretrees):

    for core in list(serialtrees.keys()):
        column = np.where(coretrees[MostBoundID_Coretag][0]==core)[0][0] #column containing core
        count = get_column_counts(coretrees[Descendent].T[column:column+1])[0]
        for p in serialtrees[core].keys():
            if Offset not in p:  #skip offsets for now
                if p in integer_properties:
                    test = np.array_equal(serialtrees[core][p], coretrees[p].T[column][0:count])
                else:
                    test = all(np.isclose(serialtrees[core][p], coretrees[p].T[column][0:count]))
                if not test:
                    print('Mismatch in {} for serial/coretrees for core {}, column {}'.format(p, core, column))

    return

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('USAGE: %s vector/serial <# tree files to write (def=3)> <tree format (core or lgal)> <Total # files for all trees (def=10000) (vector only)> <print_int (def=1000) (serial only)> <ncore_min (def=0) (serial only)> <ncore_max (def=67770) (serial only) <sim type (def=SV)>' % sys.argv[0])
    else:
        main(sys.argv)
