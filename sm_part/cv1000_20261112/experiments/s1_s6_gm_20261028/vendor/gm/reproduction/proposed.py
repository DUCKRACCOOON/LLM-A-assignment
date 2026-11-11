"""Validity-gated LLM guidance, motivated by pilot collision/parse failures.

This is a modest engineering proposal, not a claim of a new search family.
The gate checks the complete proposed polyline before using its targets.
Rejected guidance uses the same-engine g+2h control (no new model request).
"""
import time
from .core import Geometry, llm_only, parse_output, search


def proposed_search(query, geometry=None, text='', variant='validated'):
    started=time.perf_counter()
    geometry = geometry or Geometry(query)
    assessment = llm_only(query,geometry,text)
    nodes, parse_error = parse_output(text)
    integer_targets = bool(nodes) and all(all(float(v).is_integer() for v in p) for p in nodes)
    if variant == 'validated':
        accepted = assessment['valid'] and integer_targets
        fallback = 'goal_only'
    elif variant == 'parse_only':
        # Remove the collision/endpoint gate while keeping safe parser recovery.
        accepted = parse_error is None
        fallback = 'goal_only'
    elif variant == 'astar_fallback':
        accepted = assessment['valid'] and integer_targets
        fallback = 'astar'
    else:
        raise ValueError('Unknown proposal ablation')
    result=search(query,geometry,'llm_astar' if accepted else fallback,text)
    result['search_seconds']=result['seconds']
    result['seconds']=time.perf_counter()-started
    result.update(guidance_accepted=accepted,gate_reason=(None if accepted else
                  assessment['error'] or ('fractional_targets' if not integer_targets else parse_error)),
                  proposal_variant=variant,executed_method='llm_astar' if accepted else fallback)
    return result
