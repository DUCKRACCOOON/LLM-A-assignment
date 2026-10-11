"""Offline behavioral probes of the unmodified upstream source.

Run from any directory: python review/check_baseline.py
Requires the repository dependencies. Uses fixed LLM responses, disables
plotting, and makes no model/API calls. Does not reproduce Table 1.
"""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source_code"))
import matplotlib
matplotlib.use("Agg")
from llmastar.pather import AStar, LLMAStar
from llmastar.env.search.plotting import Plotting
from llmastar.utils import list_parse

Plotting.animation = lambda *args, **kwargs: None


class FixedModel:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def ask(self, **kwargs):
        self.calls += 1
        return self.response


def query(**kwargs):
    value = dict(start=[2, 2], goal=[4, 4], range_x=[0, 7],
                 range_y=[0, 7], horizontal_barriers=[], vertical_barriers=[])
    value.update(kwargs)
    return value


def llm(response="Generated Path: [[2, 2], [4, 4]]"):
    planner = LLMAStar()
    planner.model = FixedModel(response)
    return planner


def run(planner, q):
    with contextlib.redirect_stdout(io.StringIO()):
        return planner.searching(q)


def error_of(callback):
    try:
        callback()
        return None
    except Exception as exc:
        return {"type": type(exc).__name__, "message": str(exc)}


def main():
    evidence = {}
    # The same query object shrinks across successive baseline calls.
    q = query()
    before = copy.deepcopy(q)
    run(AStar(), q)
    after_a = copy.deepcopy(q)
    run(llm(), q)
    evidence["input_mutation"] = {
        "before": before["range_x"], "after_astar": after_a["range_x"],
        "after_llm_astar_same_object": q["range_x"]}
    assert evidence["input_mutation"] == {
        "before": [0, 7], "after_astar": [0, 6],
        "after_llm_astar_same_object": [0, 5]}

    blocked = query(start=[2, 2], goal=[2, 4], horizontal_barriers=[[3, 0, 6]])
    evidence["unreachable"] = {
        "astar": error_of(lambda: run(AStar(), copy.deepcopy(blocked))),
        "llm_astar": error_of(lambda: run(llm('[[2,2],[2,4]]'), copy.deepcopy(blocked)))}
    assert all(x["type"] == "KeyError" for x in evidence["unreachable"].values())

    bad = "Generated Path: [[2, 2], [bad, 3]]"
    evidence["malformed_output"] = {
        "parser_return_type": type(list_parse(bad)).__name__,
        "search_error": error_of(lambda: run(llm(bad), query()))}
    assert evidence["malformed_output"]["search_error"]["type"] == "IndexError"

    planner = llm("I cannot find a path.")
    result = run(planner, query())
    evidence["missing_output"] = {
        "targets": result["llm_output"], "current_target": planner.s_target,
        "h_at_start": planner.heuristic(planner.s_start),
        "twice_goal_distance": 2 * math.dist(planner.s_start, planner.s_goal),
        "model_calls": planner.model.calls}
    assert planner.heuristic(planner.s_start) == 2 * math.dist(planner.s_start, planner.s_goal)

    evidence["start_equals_goal"] = error_of(
        lambda: run(llm("[[3,3]]"), query(start=[3, 3], goal=[3, 3])))
    assert evidence["start_equals_goal"]["type"] == "IndexError"

    planner = llm()
    planner._initialize_parameters(query(horizontal_barriers=[[3, 2, 4]]))
    endpoint = (4, 3)
    evidence["barrier_endpoint"] = {
        "point": endpoint, "in_obs": endpoint in planner.obs,
        "retained": planner._filter_valid_nodes([endpoint]),
        "collision_from_below": planner.is_collision((4, 2), endpoint)}
    assert not evidence["barrier_endpoint"]["in_obs"]
    assert evidence["barrier_endpoint"]["retained"] and evidence["barrier_endpoint"]["collision_from_below"]

    planner._initialize_parameters(query())
    evidence["near_boundary_filter"] = {
        "retained": planner._filter_valid_nodes([[1, 3], [5, 3]]),
        "legal_moves": [not planner.is_collision((2, 3), (1, 3)),
                        not planner.is_collision((4, 3), (5, 3))]}
    assert evidence["near_boundary_filter"] == {"retained": [], "legal_moves": [True, True]}
    evidence["fractional_waypoint"] = planner._filter_valid_nodes([[3.5, 3.5]])
    assert evidence["fractional_waypoint"] == [(3.5, 3.5)]

    # Exercise the real wrapper with an in-process fake transport.
    # No openai method is invoked: the module-global openai object is replaced.
    import importlib
    wrapper = importlib.import_module("llmastar.model.chatgpt")
    captured = {}
    def fake_create(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"text": "mock"}]}
    original_openai = wrapper.openai
    try:
        wrapper.openai = SimpleNamespace(Completion=SimpleNamespace(create=fake_create))
        wrapper.ChatGPT("LLM-A*", "", None).ask("prompt", max_tokens=1000)
    finally:
        wrapper.openai = original_openai
    evidence["gpt_request_configuration"] = captured
    assert captured["stop"] == ["\n"]
    assert captured["model"] == "gpt-3.5-turbo-instruct"

    data = json.loads((ROOT / "source_code/dataset/environment_50_30.json").read_text())
    evidence["dataset"] = {"maps": len(data), "pairs": sum(len(x["start_goal"]) for x in data)}
    # A small real baseline example, preserving original collision code.
    m = data[0]
    first = {k: copy.deepcopy(m[k]) for k in ["range_x", "range_y", "horizontal_barriers", "vertical_barriers"]}
    first.update(start=m["start_goal"][0][0], goal=m["start_goal"][0][1])
    a = AStar()
    result = run(a, first)
    path = a.extract_path(a.PARENT)
    evidence["dataset_map0_pair0_astar"] = {
        "metrics": result, "unique_popped_states": len(set(a.CLOSED)),
        "duplicate_pops": len(a.CLOSED) - len(set(a.CLOSED)),
        "infinite_g_entries": sum(math.isinf(v) for v in a.g.values()),
        "path_first": path[0], "path_last": path[-1],
        "all_path_segments_collision_free": all(not a.is_collision(x, y) for x, y in zip(path, path[1:]))}
    assert result["operation"] == len(a.CLOSED)
    assert result["storage"] == len(a.g)
    assert path[0] == a.s_goal and path[-1] == a.s_start
    assert evidence["dataset_map0_pair0_astar"]["all_path_segments_collision_free"]

    # Positive example using the paper's example targets, not a live LLM output.
    guided = llm('[[5,5],[26,9],[25,23],[20,20]]')
    fresh = {k: copy.deepcopy(m[k]) for k in ["range_x", "range_y", "horizontal_barriers", "vertical_barriers"]}
    fresh.update(start=m["start_goal"][0][0], goal=m["start_goal"][0][1])
    guided_result = run(guided, fresh)
    guided_path = guided.extract_path(guided.PARENT)
    evidence["dataset_map0_pair0_fixed_waypoints"] = {
        "source": "manually supplied example waypoints; no live LLM",
        "metrics": guided_result, "model_calls": guided.model.calls,
        "path_first": guided_path[0], "path_last": guided_path[-1],
        "all_path_segments_collision_free": all(not guided.is_collision(x, y) for x, y in zip(guided_path, guided_path[1:]))}
    assert guided_path[0] == guided.s_start and guided_path[-1] == guided.s_goal
    assert guided.model.calls == 1
    assert evidence["dataset_map0_pair0_fixed_waypoints"]["all_path_segments_collision_free"]

    out = Path(__file__).with_name("baseline_evidence.json")
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    print("All behavioral probes passed; these confirm existing behavior, not correctness.")


if __name__ == "__main__":
    main()
