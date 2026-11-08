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


def map_receivers(mask, max_draws=5, prevent_reciprocal=False):
    """
    Map givers to receivers.

    Args:
        mask: Square boolean matrix of acceptable pairings.
        max_draws: Maximum attempts for standard assignment mode.
        prevent_reciprocal: If True, prohibit reciprocal exchanges
                            (A -> B and B -> A).

    Returns:
        Array where receiver_map[giver] = receiver.
    """

    if not (isinstance(mask, np.ndarray) and mask.dtype == np.bool_):
        raise TypeError("Input must be a binary np.array")

    if len(mask.shape) != 2 or mask.shape[0] != mask.shape[1]:
        raise AttributeError("Input must be a square matrix")

    N = mask.shape[0]

    if prevent_reciprocal:
        receiver_map = np.full(N, -1, dtype=int)
        available = set(range(N))

        def assign(unassigned, available):
            if not unassigned:
                return True

            # Select the most constrained remaining giver
            candidates = {}

            for giver in unassigned:
                valid = [
                    r for r in available
                    if mask[giver, r]
                    and not (
                        receiver_map[r] == giver
                    )
                ]
                candidates[giver] = valid

            giver = min(candidates, key=lambda g: len(candidates[g]))
            receivers = candidates[giver]

            if not receivers:
                return False

            # Randomize receiver selection
            np.random.shuffle(receivers)

            for receiver in receivers:
                receiver_map[giver] = receiver

                if assign(
                    unassigned - {giver},
                    available - {receiver}
                ):
                    return True

                receiver_map[giver] = -1

            return False

        if not assign(set(range(N)), available):
            raise Exception(
                "Could not identify a valid mapping. "
                "Consider fewer restrictions or more participants."
            )

        return receiver_map

    # Original assignment algorithm
    pick_order = np.argsort(np.sum(mask, axis=0))

    for attempt in range(max_draws):
        wts = np.random.uniform(low=0.0, high=1.0, size=mask.shape)
        draw = wts * mask

        receiver_map = np.zeros(N, dtype=int)
        success = True

        for giver in pick_order:
            receiver = np.argmax(draw[giver, :])

            if draw[giver, receiver] == 0.:
                success = False
                break

            receiver_map[giver] = receiver
            draw[:, receiver] = 0.

        if success:
            break

        print("draw %d failed..." % attempt)

    if not success:
        raise Exception(
            "Could not identify giver-receiver mapping. "
            "Consider fewer groups."
        )

    if not np.array_equal(np.sort(receiver_map), np.arange(N)):
        raise Exception("Algorithmic error. Givers and receivers not uniquely paired!")

    return receiver_map



def draw_and_print(hat: MagicHat, prevent_reciprocal=False):

    mask = hat.mask

    receiver_map = map_receivers(
        mask,
        prevent_reciprocal=prevent_reciprocal
    )

    NameEnum = hat.NameEnum
    max_length = len(max(hat.names, key=len))

    print("\n Assignments:")

    for giver in range(len(receiver_map)):
        receiver = receiver_map[giver]

        print("   %s is giving to %s" % (
            NameEnum(giver).name.rjust(max_length),
            NameEnum(receiver).name
        ))

    return receiver_map

