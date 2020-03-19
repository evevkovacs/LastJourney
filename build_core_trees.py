import sys
import os
import h5py
import re
import psutil
import glob
import numpy as np
import numba
from time import time
import struct
from struct import pack
from struct import unpack, calcsize

#cc_template = '09_03_2019.AQ.{}.corepropertiesextend.hdf5'
cc_template = '{}.corepropertiesextend.hdf5'
binfile_template = 'trees_099.{}'
coredir = '../CoreCatalogs'
#treedir = '../CoreTrees'
#treedir = '../CoreTrees/test'
#treedir = '../CoreTrees/new_snaps'
#treedir = '../CoreTrees/relative'
treedir = '../CoreTrees/fof_group'
#mkey = 'm_evolved_0.8_0.02'
mkey = 'm_evolved_0.9_0.005'
outfile_template = re.sub('propertiesextend', 'trees', cc_template)

coretag = 'core_tag'
foftag = 'fof_halo_tag'
coremass = 'coremass'
infall_fof_mass = 'infall_fof_halo_mass'
infall_tree_node_mass = 'infall_tree_node_mass'
first_snap = 499
last_snap = 43
#last_snap = 475
first_row = 0
last_row = 99 
#last_row = 2 
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
Zero = 'Zero'

# properties for storing in dict or matrices
core_pointers = [Descendent, DescendentOffset, FirstProgenitor, NextProgenitor,
                 FirstProgenitorOffset, NextProgenitorOffset]
#core_pointers = [Descendent, DescendentOffset, FirstProgenitor, FirstProgenitorOffset]
sibling_pointers = [FirstHaloInFOFGroup, NextHaloInFOFGroup, FirstHaloInFOFGroupOffset, NextHaloInFOFGroupOffset]
#sibling_pointers = []
core_properties_float = {'Pos_x':'x', 'Pos_y':'y', 'Pos_z':'z',
                         'Vel_x':'vx', 'Vel_y':'vy', 'Vel_z':'vz',
                         'VelDisp':'vel_disp',
                         'Vmax': 'infall_fof_halo_max_cir_vel',
                         'M_Crit200': infall_fof_mass,
                         'M_Mean200': Zero,
                         'M_TopHat': Zero,
                         'Spin_x': 'infall_sod_halo_angmom_x',
                         'Spin_y': 'infall_sod_halo_angmom_y',
                         'Spin_z': 'infall_sod_halo_angmom_z',
                         'SubHalfMass': Zero,
                        }

core_properties_int = {MostBoundID_Coretag:'core_tag',
                       'FileNr': Zero,
                       SubhaloIndex: Zero,
                      }

derived_properties_int = ['SnapNum', Len, ParentHaloTag]
#derived_properties_int = [Len]

# vector properties for storing in matrix
integer_properties = core_pointers + sibling_pointers + derived_properties_int + list(core_properties_int.keys())
float_properties = list(core_properties_float.keys())
#float_properties = ['M_Crit200']

#serial properties for storing in dict
core_properties = dict(zip(float_properties, [core_properties_float[f] for f in float_properties]))
core_properties.update(core_properties_int)
derived_properties = derived_properties_int

no_int = -999
no_float = -999.
particle_mass = 1.15e9 #M_sun/h

"""
struct halo_data
{
#    /* merger tree pointers */
    int DescendentOffset;
    int FirstProgenitorOffset;
    int NextProgenitorOffset;
    int FirstHaloInFOFgroupOffset;
    int NextHaloInFOFgroupOffset; /* properties of halo */
    int Len;
    float M_Mean200, M_Crit200, M_TopHat;
    float Pos[3];
    float Vel[3];
    float VelDisp;
    float Vmax;
    float Spin[3];
    long long MostBoundID; /* original position in subfind output */
    int SnapNum;
    int FileNr;
    int SubhaloIndex;
    float SubHalfMass;
}
"""
# header file contains Ntrees, totNHalos, TreeNHalos
header_format = "<{}i"

struct_keys = [DescendentOffset, FirstProgenitorOffset, NextProgenitorOffset,
               FirstHaloInFOFGroupOffset, NextHaloInFOFGroupOffset,
               Len, 'M_Mean200', 'M_Crit200', 'M_TopHat',
               'Pos_x', 'Pos_y', 'Pos_z', 'Vel_x', 'Vel_y', 'Vel_z',
               'VelDisp', 'Vmax', 'Spin_x', 'Spin_y', 'Spin_z',
               MostBoundID_Coretag, 'SnapNum', 'FileNr',
               SubhaloIndex, 'SubHalfMass',
              ]
struct_format = "<iiiiiiffffffffffffffqiiif"
"""
#test
struct_keys = [DescendentOffset, FirstProgenitorOffset, NextProgenitorOffset,
               FirstHaloInFOFGroupOffset, NextHaloInFOFGroupOffset,
               Len, 'M_Crit200', MostBoundID_Coretag,
#               'SnapNum', SubhaloIndex,
              ] 

#struct_format = "<iiiiiifqii"
struct_format = "<iiiiiifq"
"""

# cc = build_core_trees.get_core_snapshot( '../CoreCatalogs', snapshot)
def get_core_snapshot(coredir, snapshot, template=cc_template):
    fn = os.path.join(coredir, cc_template.format(snapshot))
    data= {}
    if os.path.exists(fn):
        h5 = h5py.File(fn, 'r')
        coredata = h5['coredata']
        keys = [k for k in list(coredata.keys()) if 'm_evolved' not in k]    
        for k in keys + [mkey]:
            data[k] = coredata[k][()]
    else:
        print('{} not found'.format(fn))

    return data

def add_coremass_column(corecat):
    mask = (corecat['central']==1) # get centrals
    central_mass = corecat[infall_tree_node_mass][mask] # get fof mass
    corecat[coremass] = corecat[mkey] # get evolved masses
    corecat[coremass][mask] = central_mass
    return corecat
    
def clean(corecat, sorted_coretags):
    mask = np.in1d(corecat[coretag], sorted_coretags, assume_unique=True)
    print('Truncating core catlog to {}/{} entries'.format(np.count_nonzero(mask),
                                                           len(corecat[coretag])))
    for p in list(corecat.keys()):
        corecat[p] = corecat[p][mask]

    return corecat

def add_snapshot_to_trees(coretrees, corecat, current_snap, snap_index=0,
                          print_int=100, coretags_fofm=None, argsorted_fofm=None,
                          sorted_indices=None,
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
                                               snap_index, current_snap,
                                               first_siblings, next_siblings, foftags_fofm_this)
            atimes.append((time()-atime1)/60.)
            #fill row of property matrix with values for selected entries
            v[snap_index][locations_this] = prop_values

        # fix first progenitors
        coretrees[FirstProgenitor] = fix_firstprogenitor_vector(coretrees[FirstProgenitor], snap_index)
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
                    coretrees[s] = add_properties_to_tree(s, loc, coretrees[s], corecat,
                                                          next_sibling, siblings[0],
                                                          snap_index, current_snap, p)
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

def get_ordered_property(p, corecat, sorted_indices_this, row, current_snap,
                         first_siblings, next_siblings, parent_tags):
    ncores = len(corecat[coretag]) # = len(sorted_indices_this)
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
    elif 'SnapNum' in p:
        #prop_values = np.array([current_snap]*ncores)
        prop_values = np.array([last_row - row]*ncores) #L-galaxies needs consecutive integers
    elif 'FileNr' in p:
        prop_values = np.array([0]*ncores)
    elif ParentHaloTag in p:
        prop_values = parent_tags
    elif Len in p:
         prop_values = (corecat[coremass][sorted_indices_this]/particle_mass).astype(int) #truncate
    elif p in list(core_properties_int.keys()):
        if Zero in core_properties_int[p]:
            prop_values = np.zeros(ncores).astype(int)
        else:
            prop_values = corecat[core_properties_int[p]][sorted_indices_this] # reorder into sorted coretag order
    elif p in list(core_properties_float.keys()):
        if Zero in core_properties_float[p]:  #zero values
            prop_values = np.zeros(ncores)
        else:
            prop_values = corecat[core_properties_float[p]][sorted_indices_this] # reorder into sorted coretag order
            if 'M_Crit' in p:
                mask = (corecat['central'][sorted_indices_this]==0) # select non-centrals
                prop_values[mask] = 0.   # set staellite masses to 0.
            if 'Spin' in p:
                prop_values /= corecat[infall_fof_mass][sorted_indices_this]
                
    else:
        print('Unknown property {}'.format(p))

    return prop_values


# fix serial trees first progenitors
#@numba.jit(nopython=True)
def fix_first_progenitors(coretrees, coretags_not_this, snap_index):

    for s in coretags_not_this:  #loop thru trees without entry in this snapshot
        if len(coretrees[s][FirstProgenitor]) == snap_index: #but have entry in last snapshot
            coretrees[s][FirstProgenitor][-1] = -1 #overwrite last element in list
            coretrees[s][FirstProgenitorOffset][-1] = -1
            
    return coretrees

#@numba.jit(nopython=True)
def add_properties_to_tree(core_tag, location, coretree, corecat,
                           next_sibling, first_sibling, snap_index,
                           current_snap, parent_fof_tag):
    if not coretree:  # empty - first entry
        coretree[Descendent] = [-1] # no descendent for first halo
        coretree[DescendentOffset] = [-1] # no descendent for first halo
        coretree[FirstProgenitor] = [1] # will be next element (if it exists)
        coretree[NextProgenitor] = [-1] # no Next_Progenitor since cores are unmerged
        coretree[FirstProgenitorOffset] = [1] # will be next element (if it exists)
        coretree[NextProgenitorOffset] = [-1] # no Next_Progenitor since cores are unmerged
        
        # initialize empty lists for properties
        for p in sibling_pointers + list(core_properties.keys()) + derived_properties:
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
    coretree[FirstHaloInFOFGroupOffset].append(first_sibling)
    coretree[NextHaloInFOFGroupOffset].append(next_sibling)
    # other properties
    if 'SnapNum' in derived_properties:
        #coretree['SnapNum'].append(current_snap)
        coretree['SnapNum'].append(last_row - snap_index)
    if ParentHaloTag in derived_properties:  
        coretree[ParentHaloTag].append(parent_fof_tag)
    coretree[Len].append(int(corecat[coremass][location]/particle_mass)) #truncate
    
    for p, v in core_properties.items():
        if Zero in v:
            if p in core_properties_int:
                coretree[p].append(0)
            else:
                coretree[p].append(0.)
        else:
            coretree[p].append(corecat[v][location])
            if 'M_Crit' in p and corecat['central'][location] == 0:
                coretree[p][-1] = 0.  #overwrite last entry with zero for satellite
            if 'Spin' in p:
                coretree[p][-1] /= corecat[infall_fof_mass][location] # divide by mass
                
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
            print('Not available yet')
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

    return trees

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
        print('Wrote header for {} fof trees with {} halos'.format(Ntrees, totNHalos))
        
        if vector:
            assert np.array_equal(foftags_to_write, coretrees[ParentHaloTag][0, start:end]), "ParentHaloTags not in fofm order"
            validate_trees(column_counts, coretrees, start, end)
            #TODO validate parent tags and FirstHaloTags (order will be same but tags will not)
            
            # write struct for each tree (ie forest) (cores in same parent halo at z=0)
            column = start
            for fof_count in fof_counts[fof_group_order]:
                offset = 0   #reset offset for each tree
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
    #add offset to coretree column if element != -1
    for p in [Descendent, FirstProgenitor]: #, NextProgenitor]:
        mask = coretrees[p][:, col] > -1
        coretrees[p + Offset][:, col][mask] += offset
        
    return coretrees

def add_offsets_serial(coretree, offset):
    #add offset to coretree if element != -1
    for p in [Descendent, FirstProgenitor]: #, NextProgenitor]:
        pvals = np.asarray(coretree[p])
        mask = (pvals > -1)
        pvals[mask] += offset
        coretree[p + Offset] = pvals

    return coretree

def replace_sibling_addresses(coretrees, col, column_counts, start, first_column_in_tree):
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
                    assert loc >= first_column_in_tree, "Sibling location is outside of tree"
                    offset = np.sum(column_counts[first_column_in_tree:loc])
                    sib_mask = (inverse==ns) & mask
                    coretrees[p + Offset][:, col][0:column_counts[cindx]][sib_mask] = offset + rows[sib_mask]
                
    return coretrees

def replace_sibling_addresses_serial(core, count, coretrees, cores_to_write, counts):
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
    vector = True if argv.get(1, 'vector')=='vector' else False
    nfiles = int(argv.get(2, 3))   #number of files to write
    print_int = int(argv.get(3, 1000)) #for serial code
    ncore_min = int(argv.get(4, 0))    #for serial code
    ncore_max = int(argv.get(5, 67760)) #for serial code (number of cores)/100
    Nfiles = int(argv.get(6, 10000)) #total number of files for vector code
    name = argv.get(0, 'test')
    print('Outputs written to {}'.format(treedir))
    
    process = psutil.Process(os.getpid())
    
    corefiles = glob.glob(coredir+'/*')
    snapshots = sorted([int(os.path.basename(f).split('.')[0]) for f in corefiles], reverse=True)  
    coretrees = {}

    for n, s in enumerate(snapshots): #process in descending order
        print('Processing snapshot {}'.format(s))
        stime = time() 
        corecat = get_core_snapshot(coredir, int(s))
        if corecat:
            corecat = add_coremass_column(corecat)
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
                    for p in integer_properties:
                        coretrees[p] = np.array([no_int]*Ncores*len(snapshots)).reshape(len(snapshots), Ncores)
                    for p in float_properties:
                        coretrees[p] = np.array([no_float]*Ncores*len(snapshots)).reshape(len(snapshots), Ncores)
                    print('Time to create arrays = {:.2f} minutes'.format((time() - ctime)/60.))
                    print('Keys are: {}'.format(sorted(list(coretrees.keys()))))
                else:
                    # check ncore_max lands on parent boundary; need to check ncore_min if != 0
                    ncore_max = get_parent_boundary(foftags_fofm, ncore_max, Ncores, name='ncore_max')
            else:
                # clean corecat to remove cores that disappeared before step 499
                corecat = clean(corecat, sorted_coretags)
            
            coretrees, atimes, stimes = add_snapshot_to_trees(coretrees, corecat, int(s), snap_index=n,
                                                              coretags_fofm=coretags_fofm,
                                                              argsorted_fofm=argsorted_coretags_fofm,
                                                              sorted_indices=indices_fofm,
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

    # TODO merging
    
    # output core trees
    numcores = Ncores if vector else len(list(coretrees.keys()))
    numfiles = Nfiles if vector else nfiles
    stride = int(np.ceil(numcores/numfiles))
    start = 0 if vector else ncore_min
    mode = '.vector' if vector else '.serial'
    print('Writing subsets of {} cores in {} files with stride {}'.format(numcores, nfiles, stride))
    
    for n in range(nfiles):
        fn = os.path.join(treedir, outfile_template.format(n)+mode)
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
            write_binary(fn_bin, coretrees, cores_to_write, foftags_to_write,
                         vector=vector, start=start, end=end, column_counts=column_counts)
        start = end
    
    #return coretrees #for testing
    # eg to run: serialtrees = build_core_trees.main({0:'test', 1:'serial', 2:2, 3:500, 4:0, 5:500})
    # coretrees = build_core_trees.main({0:'test', 1:'vector', 2:2})
    # testtree = build_core_trees.main({0:'test', 1:'serial', 2:1, 3:400, 4:246, 5:247})
    if 'test' in name:
        return coretrees
    else:
        return


MyGreen='cc'
MyBlue='ee'
MyRed='dd'
ColorScale=255
def drawforest(tree, filetype='.png'):
    treegraph=pydot.Dot(graph_type='digraph', compound='true', rankdir="TB",
                        labelloc='c', labeljust='c', ranksep=1, style="filled")
    objects = {}
    clusters =[]
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
        print('USAGE: %s vector/serial <# tree files to write (def=3)> <print_int (def=1000) (serial only)> <ncore_min (def=0) (serial only)> <ncore_max (def=67770) (serial only) <Total # files for all trees (def=10000) (vector only)>' % sys.argv[0])
    else:
        main(sys.argv)
