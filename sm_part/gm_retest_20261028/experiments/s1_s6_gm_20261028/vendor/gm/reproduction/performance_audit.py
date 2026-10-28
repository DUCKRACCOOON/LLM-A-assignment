"""Audit recorded performance without new model calls or parameter tuning."""
import json
import math
from pathlib import Path
from .diagnostics import bootstrap_map_ratio
from .experiment import ROOT, records


def compare(rows, comparator):
    result = {}
    for metric in ['operation', 'storage', 'length']:
        pairs = [(r, comparator[r['case_id']]) for r in rows
                 if r['valid'] and comparator[r['case_id']]['valid']]
        counts = {'better': 0, 'equal': 0, 'worse': 0}
        for row, other in pairs:
            a, b = row[metric], other[metric]
            key = 'equal' if math.isclose(a, b, rel_tol=1e-10) else ('better' if a < b else 'worse')
            counts[key] += 1
        result[metric] = dict(**counts, **bootstrap_map_ratio(rows, comparator, metric))
    return result


def main():
    base = records(ROOT / 'results/main/evaluations.jsonl')
    proposed = records(ROOT / 'results/main/proposed.jsonl')
    model = 'gpt-4o-mini-2024-07-18'
    selected = [r for r in proposed if r['method'] == 'validated'
                and r['model'] == model and r['prompt_style'] == 'standard']
    original = {r['case_id']: r for r in base if r['method'] == 'llm_astar'
                and r['model'] == model and r['prompt_style'] == 'standard'}
    control = {r['case_id']: r for r in base if r['method'] == 'goal_only'}
    # Exclude the whole pilot maps, not just the 30 pilot queries.
    excluded = [r for r in selected if r['map_index'] % 10 != 0]
    new = records(ROOT / 'results/generalization/evaluations.jsonl')
    result = dict(
        scope='Post-hoc audit of saved records; no new generations or algorithm changes.',
        primary_model=model, prompt='standard', n=len(selected),
        against_original=compare(selected, original),
        against_goal_only=compare(selected, control),
        excluding_pilot_maps=dict(n=len(excluded), comparison=compare(excluded, original),
                                 warning='A retrospective subset, not a newly collected blind test.'),
        new_maps=compare([r for r in new if r['method'] == 'validated'],
                         {r['case_id']: r for r in new if r['method'] == 'llm_astar'}),
        judgment=dict(released_search_fidelity='Supported by upstream parity and independent path checks.',
                      original_model_coverage='Incomplete: Llama3 absent; model substitution disclosed.',
                      public_performance='V-Gate improves over original LLM-A*, not over the g+2h control.',
                      hidden_score='Unknown: hidden data and scoring weights unavailable.',
                      next_evidence='Freeze the method before a larger independent new-map evaluation.'),
        llama3=dict(model_id='meta-llama/Meta-Llama-3-8B-Instruct', dtype='bfloat16',
                    source_max_new_tokens=8000, source_do_sample=False,
                    prompt_family='llama_prompt, not gpt_prompt', measured=False,
                    local_gpu='Intel UHD Graphics 730; no CUDA device',
                    local_ram_gib=15.6, local_torch='2.12.1+cpu',
                    responsibility='Teammate will run Llama3; user explicitly excluded it from this workspace task.',
                    status='Not run here; teammate results not yet received or merged.'),
    )
    path = ROOT / 'results/verification/performance_audit.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['n','against_original','against_goal_only','excluding_pilot_maps','new_maps']},ensure_ascii=False))


if __name__ == '__main__':
    main()
