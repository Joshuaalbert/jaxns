from typing import Dict

_PRIOR_CHAINS: Dict[int, "PriorChain"] = dict() # all the prior chains currently in existence
_PRIOR_CHAIN_NEXT_INDEX = 0  # which prior chain we are on
_PRIOR_CHAIN_INDEX_STACK = [] # stack of indicies (depth)