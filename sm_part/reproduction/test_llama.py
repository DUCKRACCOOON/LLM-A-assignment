"""Offline safeguards; synthetic fixtures never enter experiment results."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import llama_run as runner
from llama_report import summarize, check_baselines


class LlamaProtocolTests(unittest.TestCase):
    def test_scope_and_model_specific_prompts(self):
        from core import PROMPTS as gpt_prompts
        tasks = runner.tasks_for('main')
        self.assertEqual(len(tasks), 3000)
        self.assertEqual(len({(q['id'], k) for q, k in tasks}), 3000)
        self.assertEqual(len(runner.tasks_for('pilot')), 30)
        self.assertFalse({q['id'] for q, _ in tasks} & {q['id'] for q, _ in runner.tasks_for('pilot')})
        for kind in runner.KINDS:
            self.assertIn('<|start_header_id|>', runner.PROMPTS[kind])
            self.assertNotEqual(runner.PROMPTS[kind], gpt_prompts[kind])

    def test_context_limits_reject_truncated_inputs(self):
        self.assertEqual(runner.token_budget(2000, 8192), 6192)
        self.assertEqual(runner.token_budget(100, 8192), 8000)
        self.assertEqual(runner.token_budget(8191, 8192), 1)
        with self.assertRaises(RuntimeError):
            runner.token_budget(8192, 8192)

    def test_resume_rejects_changed_request_and_corrupt_result(self):
        q = runner.cases_for('pilot')[0]
        result = {'content': '[[3,4],[43,25]]'}
        record = {'case_id': q['id'], 'prompt_kind': 'standard', 'run_id': 'test',
                  'request_sha256': runner.digest({'case': q, 'prompt': runner.PROMPTS['standard'].format(**q), 'run_id': 'test'}),
                  'result': result, 'result_sha256': runner.digest(result)}
        runner.check_record(record, 'test', q, 'standard')
        with self.assertRaises(RuntimeError):
            runner.check_record(record, 'changed-model', q, 'standard')
        corrupt = copy.deepcopy(record)
        corrupt['result']['content'] += 'changed'
        with self.assertRaises(RuntimeError):
            runner.check_record(corrupt, 'test', q, 'standard')

    def test_missing_responses_never_become_complete(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(runner, 'OUT', Path(folder)):
            runner.save(Path(folder) / 'pilot/manifest.json', {
                'cases': runner.cases_for('pilot'), 'identity': {'protocol': runner.protocol()}})
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                runner.evaluate('pilot')

    def test_invalid_paths_keep_full_validity_denominator(self):
        base = [{'case_id': 'a', 'operation': 10, 'storage': 10, 'length': 10, 'valid': True},
                {'case_id': 'b', 'operation': 10, 'storage': 10, 'length': 10, 'valid': True}]
        rows = [{'case_id': 'a', 'method': 'llm_only', 'prompt': 'standard', 'valid': True,
                 'operation': None, 'storage': None, 'length': 20},
                {'case_id': 'b', 'method': 'llm_only', 'prompt': 'standard', 'valid': False,
                 'operation': None, 'storage': None, 'length': 5, 'failure': 'collision'}]
        result = summarize(rows, base)[0]
        self.assertEqual(result['valid_ratio'], 50)
        self.assertEqual(result['length_ratio'], 200)
        self.assertEqual(result['length_n'], 1)
        self.assertAlmostEqual(result['length_all_parseable_ratio'], 100)
        self.assertEqual(result['length_all_parseable_n'], 2)

    def test_baseline_mismatch_blocks_combined_report(self):
        base = [{'case_id': 'a', 'operation': 10, 'storage': 10, 'length': 10, 'valid': True}]
        changed = copy.deepcopy(base)
        changed[0]['operation'] = 9
        with self.assertRaisesRegex(RuntimeError, 'mismatch'):
            check_baselines(changed, base)


if __name__ == '__main__':
    unittest.main()
