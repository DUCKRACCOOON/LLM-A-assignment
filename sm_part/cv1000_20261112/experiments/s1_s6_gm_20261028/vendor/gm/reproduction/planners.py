"""Assignment-facing searching(query, filepath=...) adapters.

Structured input follows upstream. A natural-language string is parsed through
the upstream parsing prompt. Set model/prompt at construction, never per map.
"""
import json
import runpy
from .api import completion
from .core import Geometry, search
from .experiment import PROMPT_FILE
from .proposed import proposed_search


def model_text(model,prompt):
    response=completion(dict(model=model,messages=[dict(role='user',content=prompt)],
                             temperature=0,max_tokens=1000,store=False))
    if response['status']!='ok':
        raise RuntimeError('LLM request failed; inspect account/model availability without exposing credentials.')
    return response['response']['choices'][0]['message'].get('content') or ''


def normalize_query(query,model):
    if isinstance(query,str):
        templates=runpy.run_path(str(PROMPT_FILE))
        prompt=templates['sysprompt_parse']
        examples=templates['example_parse']
        prompt+='\nFollow these examples delimited with “”" as a guide.\n'
        for example in examples.values():
            split=example.find('\n')+1
            prompt+=f'“”"\nUser: {example[:split-1]}\nAssistant: {example[split:]}“”"\n'
        query=json.loads(model_text(model,prompt+'\nUser: '+query))
    required=['start','goal','range_x','range_y','horizontal_barriers','vertical_barriers']
    if not isinstance(query,dict) or any(k not in query for k in required):
        raise ValueError('Expected upstream query keys: '+', '.join(required))
    for key in ['start','goal','range_x','range_y']:
        if len(query[key])!=2 or any(type(x)!=int for x in query[key]):
            raise ValueError(key+' must contain two integers')
    for key in ['horizontal_barriers','vertical_barriers']:
        if any(len(barrier)!=3 or any(type(x)!=int for x in barrier) for barrier in query[key]):
            raise ValueError(key+' must contain integer triples')
    return query


def save_plot(query,result,filepath):
    if filepath is None:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for y,a,b in query['horizontal_barriers']:ax.plot([a,b],[y,y],'k-',lw=2)
    for x,a,b in query['vertical_barriers']:ax.plot([x,x],[a,b],'k-',lw=2)
    if result.get('path'):ax.plot(*zip(*result['path']),color='#2364aa',lw=2)
    ax.scatter(*query['start'],marker='s',color='green',s=60)
    ax.scatter(*query['goal'],marker='*',color='red',s=100)
    ax.set(xlim=(query['range_x'][0],query['range_x'][1]-1),ylim=(query['range_y'][0],query['range_y'][1]-1),aspect='equal')
    ax.set_title('Valid path' if result['valid'] else result.get('error','No path'))
    fig.tight_layout();fig.savefig(filepath,dpi=150);plt.close(fig)


class AStar:
    def searching(self,query,filepath='temp.png'):
        query=normalize_query(query,'gpt-4o-mini')
        result=search(query,method='astar')
        save_plot(query,result,filepath)
        return result


class LLMAStar:
    def __init__(self,llm='gpt',prompt='standard',model='gpt-4o-mini',provider=None):
        if llm!='gpt':
            raise ValueError('This reproduction provides an OpenAI adapter; configure model explicitly.')
        if prompt not in ['standard','cot','repe']:
            raise ValueError('Unknown prompt')
        self.prompt,self.model,self.provider=prompt,model,provider or model_text

    def searching(self,query,filepath='temp.png'):
        query=normalize_query(query,self.model)
        template=runpy.run_path(str(PROMPT_FILE))['gpt_prompt'][self.prompt]
        prompt=template.format(start=query['start'],goal=query['goal'],
                               horizontal_barriers=query['horizontal_barriers'],vertical_barriers=query['vertical_barriers'])
        response=self.provider(self.model,prompt)
        result=self._search(query,response)
        result['llm_output']=result.get('targets',[])
        save_plot(query,result,filepath)
        return result

    def _search(self,query,response):
        return search(query,method='llm_astar',text=response)


class Proposed(LLMAStar):
    def _search(self,query,response):
        return proposed_search(query,text=response)
