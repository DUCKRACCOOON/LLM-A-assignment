import copy
import random
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from core import load_cases, search, parse_path, validate
from run import cases_for, payload
from improve_core import refine, diagnosis, correction_payload, delayed_search

class ImprovementTests(unittest.TestCase):
    def test_search_equivalence(self):
        for q in random.Random(240702511).sample(load_cases(),30):
            for raw in ([], [tuple(q['start']),(26,9),(25,23),tuple(q['goal'])]):
                a = search(q,'llm_astar',raw)
                b = delayed_search(q,lambda:raw,mode='immediate')
                for k in ('operation','storage','length','valid','path','targets'):
                    self.assertEqual(a[k],b[k],(q['id'],k))
            a = search(q,'llm_astar',[])
            b = delayed_search(q,lambda:self.fail('never callback'),mode='never')
            for k in ('operation','storage','length','valid','path','targets'):
                self.assertEqual(a[k],b[k])

    def test_adaptive_shared_response_replay(self):
        calls = 0
        detour=dict(id='unit-detour',start=[5,5],goal=[15,5],range_x=[0,21],range_y=[0,21],
                    horizontal_barriers=[],vertical_barriers=[[10,0,18]])
        for q in cases_for('pilot')+[detour]:
            before = copy.deepcopy(q); count = []
            def get(): count.append(1); return [tuple(q['start']),(26,13),tuple(q['goal'])]
            a=delayed_search(q,get)
            b=delayed_search(q,lambda:[tuple(q['start']),(26,13),tuple(q['goal'])])
            self.assertLessEqual(len(count),1)
            self.assertEqual(len(count),a['api_calls'])
            for k in ('operation','storage','length','path','trigger'): self.assertEqual(a[k],b[k])
            self.assertEqual(q,before); self.assertTrue(a['valid'])
            calls += a['api_calls']
        self.assertGreater(calls,0)

    def test_refine_preserves_input_and_removes_loops(self):
        q=dict(start=[3,3],goal=[8,8],range_x=[0,12],range_y=[0,12],horizontal_barriers=[],vertical_barriers=[])
        raw=[(3,3),(4,4),(4,4),(5,4),(4,4),(8,8)]; before=copy.deepcopy(raw)
        got,meta=refine(q,raw)
        self.assertEqual(got,[(3,3),(8,8)]); self.assertEqual(raw,before)
        self.assertTrue(meta['accepted']); self.assertTrue(validate(q,got)[0])
        q['horizontal_barriers']=[[6,0,11]]
        got,meta=refine(q,raw)
        self.assertFalse(meta['accepted']); self.assertEqual(got,[])

    def test_feedback_only_payload_difference(self):
        q=cases_for('pilot')[0]; text=str([q['start'],q['goal']])
        self.assertFalse(diagnosis(q,text)['valid'])
        a=correction_payload(q,'standard',text,False)
        b=correction_payload(q,'standard',text,True)
        self.assertEqual({k:v for k,v in a.items() if k!='messages'},{k:v for k,v in b.items() if k!='messages'})
        prefix='\n\nExternal validation report:\n'; suffix='\n\nReview the previous response'
        def skeleton(x):
            s=x['messages'][0]['content']; return s.split(prefix)[0]+s.split(suffix)[1]
        self.assertEqual(skeleton(a),skeleton(b)); self.assertEqual(len(a['messages']),1)
        self.assertTrue(a['messages'][0]['content'].startswith(payload(q,'standard')['messages'][0]['content']))
        self.assertEqual(diagnosis(q,'No route')['error'],'missing_list')

    def test_pilot_pipeline_contract_with_fake_responses(self):
        import improve_run as ir
        import improve_report as report
        from run import save,digest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(ir,'OUT',root),patch.object(report,'OUT',root):
                runner=ir.Runner('unit-test-no-network',1)
                def api(stage,q,kind,variant,body):
                    p=root/stage/'raw'/f'{q["id"]}-{kind}-{variant}.json'
                    if p.exists(): return json.loads(p.read_text())
                    response={'model':ir.MODEL,'choices':[{'message':{'content':'No route'},'finish_reason':'stop'}],
                              'usage':{'prompt_tokens':1,'completion_tokens':1}}
                    r={'case_id':q['id'],'prompt_kind':kind,'variant':variant,'created_utc':ir.now(),
                       'payload':body,'payload_sha256':digest(body),'response':response,
                       'response_sha256':digest(response),'latency_seconds':0.0}
                    save(p,r);return r
                runner.api=api
                for q in cases_for('pilot'):
                    runner.common('pilot',q);runner.feedback('pilot',q);runner.delayed('pilot',q)
                rows,check=report.verify_stage('pilot')
                self.assertEqual(len(rows),270);self.assertEqual(check['usage']['requests'],90)
                self.assertEqual(len(report.summary(rows)),27)
                # Expected output completeness cannot silently pass after a missing artifact.
                (root/'pilot/e3/dev-09.json').unlink()
                with self.assertRaises(FileNotFoundError): report.verify_stage('pilot')

if __name__=='__main__': unittest.main(verbosity=2)
