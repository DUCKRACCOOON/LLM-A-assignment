"""Build K2 CoT/RePE templates with SM's disjoint example maps (deterministic reasoning text).

Original GM/upstream cot and repe templates contain 3 examples taken from standard-example
positions 1, 2 and 5. We keep that positional choice (SM examples index 0, 1, 4), the
original instruction header and output format; only maps, paths and reasoning text change.
Reasoning is generated from geometry: walls that block the straight line from the current
point to the goal are named, the next waypoint is reported as the bypass point.
"""
import runpy
from common import *

ORIGINAL = runpy.run_path(str(HERE/'vendor/llm-astar/llmastar/pather/llm_a_star/prompt.py'))['gpt_prompt']
EXAMPLE_POSITIONS = [0, 1, 4]   # mirrors upstream cot/repe using standard examples 1, 2, 5
ORDINALS = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth']

def wall_text(kind, w):
    if kind == 'horizontal_barriers': return f'the horizontal barrier at y={w[0]} spanning x={w[1]} to x={w[2]}'
    return f'the vertical barrier at x={w[0]} spanning y={w[1]} to y={w[2]}'

def blocking(q, a, b):
    out = []
    for kind in ('horizontal_barriers', 'vertical_barriers'):
        for w in q[kind]:
            single = dict(q, horizontal_barriers=[], vertical_barriers=[]); single[kind] = [w]
            if gm.Geometry(single).collision(tuple(a), tuple(b)): out.append(wall_text(kind, w))
    return out

def join(items): return items[0] if len(items) == 1 else ', '.join(items[:-1])+' and '+items[-1]

def header_and_tail(template):
    header = template.split('Start Point:')[0]
    tail = 'Start Point: {start}'+template.split('Start Point: {start}', 1)[1]
    return header, tail

def problem_block(q):
    return ('Start Point: '+str(q['start'])+'\nGoal Point: '+str(q['goal'])+'\nHorizontal Barriers: '+str(q['horizontal_barriers'])
            +'\nVertical Barriers: '+str(q['vertical_barriers']))

def cot_example(e):
    q, w = e['query'], [list(p) for p in e['waypoints']]; s, g = w[0], w[-1]
    if len(w) == 2:
        thought = f'There are no barriers obstructing the direct path from {s} to {g}, so move directly to {g}.'
    else:
        steps = []
        for cur, nxt in zip(w[:-2], w[1:-1]):
            walls = blocking(q, cur, g)
            steps.append(f'avoiding {join(walls)} by moving to {nxt}' if walls else f'moving to {nxt}')
        thought = f'Identify a path from {s} to {g} by '+', then '.join(steps)+f', and finally move directly to {g}.'
    return problem_block(q)+'\nThought: '+thought+'\nGenerated Path: '+json.dumps(w)

def repe_example(e):
    q, w = e['query'], [list(p) for p in e['waypoints']]; g = w[-1]; lines = []
    for i, (cur, nxt) in enumerate(zip(w, w[1:])):
        lines.append(f'- {ORDINALS[i]} Iteration on {cur}')
        walls = blocking(q, cur, g)
        if nxt == g:
            lines += ['Thought: From this position, there are no barriers directly obstructing the path to the goal.',
                      f'Selected Point: {nxt}', 'Evaluation: The path to the goal is clear from here, allowing a direct move to the goal.']
        elif walls:
            lines += [f'Thought: {join(walls)[0].upper()+join(walls)[1:]} {"blocks" if len(walls) == 1 else "block"} the direct path to the goal. To navigate around {"it" if len(walls) == 1 else "them"}, we should move past {"the end of the barrier" if len(walls) == 1 else "the ends of the barriers"}.',
                      f'Selected Point: {nxt}',
                      f'Evaluation: The selected point {nxt} effectively bypasses {join(walls)}, maintaining progress toward the goal without encountering additional obstacles.']
        else:
            lines += ['Thought: Moving to an intermediate point keeps the route clear of nearby barriers.',
                      f'Selected Point: {nxt}', f'Evaluation: The selected point {nxt} keeps the path collision-free and closer to the goal.']
    return problem_block(q)+'\n'+'\n'.join(lines)+'\nGenerated Path: '+json.dumps(w)

def build():
    examples = read(INPUTS/'prompts/examples.json'); chosen = [examples[i] for i in EXAMPLE_POSITIONS]
    for e in chosen:
        q = e['query']; assert gm.validate_path(gm.Geometry(q), [tuple(p) for p in e['waypoints']], q['start'], q['goal']) is None
    out = {}
    for name, maker in (('cot', cot_example), ('repe', repe_example)):
        header, tail = header_and_tail(ORIGINAL[name])
        assert ORIGINAL[name].count('Start Point:')-1 == len(chosen)
        text = header+'\n\n'.join(maker(e) for e in chosen)+'\n\n'+tail
        text.format(start=[0, 0], goal=[1, 1], horizontal_barriers=[], vertical_barriers=[])   # template must format
        path = RUN/'prompts'/f'{name}.txt'; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding='utf-8')
        out[name] = dict(path=path.relative_to(HERE).as_posix(), sha256=sha(path))
    save(RUN/'prompts/provenance.json', dict(created_at=now(), source_template=str((HERE/'vendor/llm-astar/llmastar/pather/llm_a_star/prompt.py').relative_to(HERE).as_posix()),
         source_template_sha256=sha(HERE/'vendor/llm-astar/llmastar/pather/llm_a_star/prompt.py'), example_source='inputs/prompts/examples.json (SM seed 20261112, disjoint from public 100 maps)',
         example_positions=EXAMPLE_POSITIONS, examples_count=len(chosen), templates=out,
         reasoning='deterministic from geometry: walls blocking current->goal straight line are named; next waypoint reported as bypass point',
         note='Upstream cot/repe use standard examples 1,2,5; same positions used here instead of first three, so detour examples are retained.'))
    return out

if __name__ == '__main__':
    print(json.dumps(build(), indent=1))
