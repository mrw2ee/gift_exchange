import numpy as np
from enum import Enum, auto
from typing import List
import itertools
import warnings

class MagicHat:
    """
    MagicHat - Class for storing list of names and groups of names which should not be paired

        Add groups of names which should not be paired. Names automatically added to the name list

        Add individual names that can be paired with anyone else

        Get binary matrix of acceptable pairings
    """

    def __init__(self):
        # Initialize an empty list to store names
        self.__names = []
        self.__groups=[]
        

    def add_name(self, name : str)->None:
        # Add a name to the list
        if not isinstance(name, str):
            raise TypeError("Input must be a string.")
        if name in self.__names:
            warnings.warn("Attempted to add name already in list: "+name)
        else:
            self.__names.append(name)

    def add_group(self, group:List[str])->None:
        # Add group of names that should not be paired
        if not isinstance(group, list):
            raise TypeError("Input must be a list of strings.")

        # Ensure all names are in the name list
        for n in group:
            if n not in self.__names:
                self.add_name(n)
        
        # Add group to the restriction list
        self.__groups.append(group)

    def __str__(self):
        # Return a string representation of the list of names
        s = ", ".join(self.__names)
        if len(self.__groups) >0:
            for g in self.__groups:
                s = s + "\n  Group: "+ ", ".join(g)
        return s

    @property
    def names(self)->List[str]:
        return self.__names
    
    @property
    def groups(self)->List[List[str]]:
        return self.__groups
    
    @property
    def NameEnum(self)->Enum:
        """
        Enumeration from name list (0-based index)
        """
        return Enum('NameEnum',{name: idx for idx, name in enumerate(self.__names)})

    @property
    def mask(self) -> np.ndarray:
        """
        Binary matrix representing potential pairings
        """
        NameEnum = self.NameEnum
        N = len(NameEnum)
        mask = np.ones((N,N), dtype=bool)
        # no self-gifting
        mask *=~np.eye(N, dtype=bool)
        for g in self.groups:
            for pair in list(itertools.combinations(g, 2)):
                id0 = NameEnum[pair[0]].value
                id1 = NameEnum[pair[1]].value
                mask[[id0,id1],[id1,id0]] = False
        return mask

def map_receivers(mask,max_draws=5):
    """
    map givers to receiver

    Accepts square binary matrix
    """

    # Check if the matrix is square
    if not (isinstance(mask,np.ndarray) and mask.dtype == np.bool_):
        raise TypeError("Input must be a binary np.array") 
    if (mask.shape[0] != mask.shape[1]) or len(mask.shape) != 2:
        raise AttributeError("Input must be a square matrix")

    # Pick for most-constrained name first...
    pick_order = np.argsort(np.sum(mask,axis=0))

    for attempt in range(max_draws):
        # Attempt to pair givers and receivers.
        #
        # For each giver, attempt to assign a receiver. It is possible that, by the time we reach the last giver, the only remaining receiver is in one of the giver's exclusion groups. If that happens, re-draw and try again.


        # Generate a matrix of uniformly distributed random values in the range [0, 1)
        wts = np.random.uniform(low=0.0, high=1.0, size=mask.shape)
        draw = wts*mask

        receiver_map = np.zeros(len(pick_order),dtype=int)
        success = True
        for giver in pick_order:
            # Get the index of the maximum value
            receiver = np.argmax(draw[giver,:])
            if draw[giver,receiver] == 0.:
                # No one left for this name to gift
                success = False
            # Assign receiver to giver
            receiver_map[giver] = receiver
            # Remove receiver from pool
            draw[:,receiver] = 0.0

        if success:
            break
        print("draw %d failed..."%attempt)

    if not success:
        raise Exception("Could not identify giver-receiver mapping. Consider fewer groups.")

    # Check 1:1 mapping. Everyone should be a unique receiver
    if not np.array_equal(np.sort(receiver_map),np.arange(0,len(receiver_map))):
        raise Exception("Algorithmic error. Givers and receivers not uniquely paired!")

    return receiver_map

def draw_and_print(hat : MagicHat):
    # Get  a binary matrix of potential giver-receiver pairs
    mask = hat.mask
    # Draw names
    map = map_receivers(mask)

    # Print result
    NameEnum = hat.NameEnum
    max_length = len(max(hat.names, key=len))
    print("\n Assignments:")
    for giver in range(len(map)):
        receiver = map[giver]
        print("   %s is giving to %s"%(NameEnum(giver).name.rjust(max_length),NameEnum(receiver).name))

    return map
