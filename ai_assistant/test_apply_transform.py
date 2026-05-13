import runpy

mod = runpy.run_path('ai_assistant/langgraph_skeleton.py')
_apply = mod['_apply_transform_plan']

data = [ {'articleCode':'A1','quantite':2},{'articleCode':'A1','quantite':3},{'articleCode':'A2','quantite':5} ]
plan = {'steps':[{'op':'aggregate','groupby':['articleCode'],'aggs':[{'field':'quantite','agg':'sum','as':'total_qty'}]},{'op':'limit','value':100}]}
res = _apply(data, plan)
print('RESULT:', res)
print('LEN', len(res))
