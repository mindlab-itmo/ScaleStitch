"""Portable CPU analysis of a fixed 3-host x 3-task x 4-condition matrix.

Metadata rows bind the ordered cohort, active object mapping, full T/C/S
packets, CPU question/batch input_fingerprint, data/cohort ledger data_sha256,
packet hashes and prefix_budgets. These input identities exclude model weights.
input_bindings[host][task][mode] repeats those identities from each verified
preflight; raw_banks has the same layout with unmodified prediction records.
Model checkpoint/export provenance is supplied separately. All 36 cells and
input-defined populations are validated before scoring any answer. The JSON
config loads recorded inputs and predictions without invoking a model.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import re
PROTOCOL_SHA='6c39b847a11828a74cf173241798b78566c4fe690b9b057f80fa74b331b617e1'
TASKS=('radius','height','sqa3d')
HOSTS=('parent','Atrain','Ttrain')
MODES=('A','T','C','S')
COUNTS=dict(radius=4161,height=4157,sqa3d=3519)
FAMILIES=dict(radius=70,height=70,sqa3d=67)
SEED,REPLICATES,FAMILY_SIZE=20260911,100000,9
DEPENDENCY_CANONICAL_SHA256={
    'text_baseline_statistics':'d5f1a1248dcba40ae9aa5c26e0f0c52f92614d6f5d6b79a9e29482f1159f399b',
    'reviewer_diagnostics':'36439106aa7e535676fe5486102add53a5323ad6806091184e6e5b68af2d2a6e',
    'host_normalization':'005a736b6e397d2287bdb0a4bc54afb6cd46cdd6e940a021f1cd4fca3b3c3c87',
    'prediction_scoring':'25f7d443e76a3bff1c3a6b42581cf7e97d0a8902fa2290f8df37abb2b988a8a7',
    'natural_qa_metrics':'74d9b2c8b771b0783ec52ca817ce215143ae8880b306bd607d71690626271387',
}
PINS={}
REFERENCE_ANALYSIS_SHA256='9098083d24cc21458e0eb6abac439eff789a8b16bbf2e70751b02030550e4e4f'
WEIGHTS={'T_by_'+mode:{'Ttrain.T':1,'Ttrain.'+mode:-1,'Atrain.T':-1,'Atrain.'+mode:1} for mode in 'ACS'}


def require(value,message):
    if not value:raise ValueError(message)


def _source_fingerprints(root):
    actual={}
    for name,expected in DEPENDENCY_CANONICAL_SHA256.items():
        data=(Path(root)/(name+'.py')).read_bytes()
        require(hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()==expected,'pinned canonical CPU source differs: '+name)
        actual[name]=hashlib.sha256(data).hexdigest()
    return actual


def dependencies():
    global PINS
    root=Path(__file__).resolve().parent
    actual=_source_fingerprints(root)
    loaded={}
    for name in DEPENDENCY_CANONICAL_SHA256:
        module=importlib.import_module(__package__+'.'+name)
        require(Path(module.__file__).resolve()==root/(name+'.py'),'CPU dependency loaded from a different package: '+name)
        loaded[name]=module
    PINS=actual
    return dict(stats=loaded['text_baseline_statistics'],diag=loaded['reviewer_diagnostics'],
        normalizer=loaded['host_normalization'],scorer=loaded['prediction_scoring'],metrics=loaded['natural_qa_metrics'])


def _hash(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _surface(row,meta,end_sym):
    """Validate the retained decoder's raw-to-source remap, without scoring it."""
    require(isinstance(row.get('raw_prediction'),str) and isinstance(row.get('prediction'),str)
        and type(row.get('invalid_object_id')) is bool and not row.get('error'),'malformed raw prediction or technical error')
    text=row['raw_prediction'].split(end_sym,1)[0].replace('  ',' ').replace(' .','.').strip()
    mapping={obj['visible_id']:obj['source_row'] for obj in meta['objects']}
    matches=list(re.finditer(r'<OBJ([^>]*)>',text))
    invalid=text.count('<OBJ')!=len(matches) or any(not re.fullmatch('[0-9]{3}',match[1])
        or int(match[1]) not in mapping for match in matches)
    if not invalid:text=re.sub(r'<OBJ([0-9]{3})>',lambda match:f'<OBJ{mapping[int(match[1])]:03d}>',text)
    require(row['prediction']==text and row['invalid_object_id']==invalid,'raw/source mapping or invalid-ID flag differs')


def _metadata(task,rows,refs,expected_count,expected_families,deps):
    require(isinstance(rows,list) and len(rows)==expected_count,'complete fixed input cohort required')
    qids=[str(row['qid']) for row in rows]
    require(len(qids)==len(set(qids)) and all(qids),'duplicate or empty metadata identity')
    require(isinstance(refs,list) and [str(row['question_id']) for row in refs]==qids,'ordered complete references differ')
    for row,reference in zip(rows,refs):
        require(re.fullmatch(r'scene[0-9]{4}_[0-9]{2}',row['scene_id']) is not None
            and row['scene_family']==row['scene_id'].split('_')[0] and reference['scene_id']==row['scene_id'],'scene/family/reference differs')
        require(isinstance(reference['answers'],list) and reference['answers']
            and all(isinstance(answer,str) and answer.strip() for answer in reference['answers']),'nonempty reference answers required')
        objects=row['objects'];mapping={obj['visible_id']:obj['source_row'] for obj in objects}
        require(len(objects)>=2 and len(mapping)==len(objects) and len(set(mapping.values()))==len(objects)
            and set(mapping)==set(mapping.values()) and all(type(v) is int and 0<=v<=999 for pair in mapping.items() for v in pair),
            'active source/visible assignment differs')
        require(_hash(row['input_fingerprint']) and _hash(row['data_sha256'])
            and set(row['packet_sha256'])==set(MODES) and row['packet_sha256']['A'] is None
            and all(_hash(row['packet_sha256'][mode]) for mode in 'TCS'),'data/question/packet source identity missing')
        require(set(row['packets'])==set(MODES) and row['packets']['A'] is None
            and set(row['prefix_budgets'])==set(MODES),'all four input packet/budget bindings required')
        sources=sorted(mapping.values());donors={source:sources[index-1] for index,source in enumerate(sources)}
        truth={item['source_row']:item['value_cm'] for item in row['packets']['T']}
        for mode in MODES:
            budget=row['prefix_budgets'][mode]
            require(set(budget)=={'native_tokens','packet_tokens','prefix_tokens','numerical_blocks'}
                and all(type(n) is int and n>=0 for n in budget.values()) and budget['native_tokens']>0
                and budget['prefix_tokens']==budget['native_tokens']+budget['packet_tokens']
                and budget['prefix_tokens']+32<=2502 and budget['native_tokens']==row['prefix_budgets']['A']['native_tokens']
                and (budget['packet_tokens']==budget['numerical_blocks']==0 if mode=='A'
                     else budget['packet_tokens']>0 and budget['numerical_blocks']==len(objects)),'prefix budget differs')
            if mode=='A':continue
            packet=row['packets'][mode]
            require([{key:item[key] for key in ('visible_id','source_row')} for item in packet]==objects,'full-scene packet mapping/order differs')
            for item in packet:
                donor=None if mode=='C' else item['source_row'] if mode=='T' else donors[item['source_row']]
                require('value_source_row' in item and item['value_source_row']==donor
                    and type(item['value_cm']) is int and item['value_cm']>0
                    and item['value_cm']==(100 if mode=='C' else truth[donor]),'full-scene donor or value differs')
        if task=='sqa3d':
            require(not {'candidate_source_rows','target_source_row','direction'}.intersection(row),'natural QA has no direct target/ranking')
        else:
            require(reference['answers']==[f"<OBJ{row['target_source_row']:03d}>"],'physical reference target differs')
    require(len({row['scene_family'] for row in rows})==expected_families,'contributing scene-family count differs')
    prepared=dict(rows={str(row['qid']):row for row in rows},qids=qids)
    if task=='sqa3d':
        deps['metrics']._references(refs,metric_names=deps['metrics'].METRICS+(deps['normalizer'].METRIC,))
    else:
        direct=deps['diag']._metadata(rows,MODES)
        for qid,item in direct.items():
            require(item['direction']=='max' and deps['diag']._winner(item['packets']['T'],item['candidates'],'max')==item['target'],
                'unique correct-packet physical winner differs')
            winner=deps['diag']._winner(item['packets']['S'],item['candidates'],'max')
            item['s_ranking']='tied' if winner is None else 'preserved' if winner==item['target'] else 'flipped'
            item['packet_winners']={mode:deps['diag']._winner(item['packets'][mode],item['candidates'],'max') for mode in MODES}
        prepared['direct']=direct
    return prepared


def validate(metadata,references,raw_banks,input_bindings,model_provenance,*,end_sym,expected_counts=None,expected_families=None):
    deps=dependencies();counts=COUNTS if expected_counts is None else expected_counts
    families=FAMILIES if expected_families is None else expected_families
    require(isinstance(end_sym,str) and end_sym,'recorded end_sym required')
    require(all(set(value)==set(TASKS) for value in (metadata,references,counts,families)),'all three complete tasks required')
    require(all(type(value) is int and value>0 for value in list(counts.values())+list(families.values())),'positive full counts/families required')
    require(all(set(value)==set(HOSTS) for value in (raw_banks,input_bindings,model_provenance)),'all three checkpoints required')
    require(all(isinstance(value,dict) and _hash(value.get('host_checkpoint_sha256')) for value in model_provenance.values()),
        'model provenance must be separate from common input bindings')
    prepared={task:_metadata(task,metadata[task],references[task],counts[task],families[task],deps) for task in TASKS}
    generation=dict(max_new_tokens=32,num_beams=1,do_sample=False,use_cache=True,repetition_penalty=1.)
    for host in HOSTS:
        require(set(raw_banks[host])==set(TASKS) and set(input_bindings[host])==set(TASKS),'missing/extra checkpoint task')
        for task in TASKS:
            require(set(raw_banks[host][task])==set(MODES) and set(input_bindings[host][task])==set(MODES),'missing/extra condition cell')
            for mode in MODES:
                predictions=raw_banks[host][task][mode];bindings=input_bindings[host][task][mode]
                wanted=[]
                for meta in metadata[task]:
                    wanted.append(dict(qid=str(meta['qid']),scene_id=meta['scene_id'],input_fingerprint=meta['input_fingerprint'],
                        data_sha256=meta['data_sha256'],packet_sha256=meta['packet_sha256'][mode],prefix_budget=meta['prefix_budgets'][mode]))
                require(bindings==wanted,'ordered data/cohort/feature/mask/assignment/question/packet/prefix bindings differ')
                require(isinstance(predictions,list) and [str(row['qid']) for row in predictions]==prepared[task]['qids'],
                    'missing, extra, duplicate or reordered raw prediction matrix')
                for row in predictions:
                    meta=prepared[task]['rows'][str(row['qid'])]
                    require(row['scene_id']==meta['scene_id'] and row['condition']==mode,'raw prediction scene/condition differs')
                    actual=row.get('effective_generation')
                    require(isinstance(actual,dict) and {key:value for key,value in actual.items() if key!='decision_mode'}==generation
                        and actual.get('decision_mode','free')=='free','raw prediction effective decoder/generation differs')
                    require(all(row.get(field)==value for field,value in meta['prefix_budgets'][mode].items()),
                        'raw prediction prefix differs from verified input budget')
                    _surface(row,meta,end_sym)
    return prepared


def score_cells(raw_banks,references,prepared,*,end_sym):
    deps=dependencies();scored={task:{} for task in TASKS}
    for host in HOSTS:
        for task in TASKS:
            if task=='sqa3d':
                _,bank=deps['scorer'].score_predictions([row for mode in MODES for row in raw_banks[host][task][mode]],
                    references[task],cohort=prepared[task]['qids'],end_sym=end_sym,scoring='host_normalized_em1')
                require(bank['primary_metric']==deps['normalizer'].METRIC,'natural-QA metric label differs')
                for mode,rows in bank['conditions'].items():
                    scored[task][host+'.'+mode]={qid:dict(row,blank=not raw['prediction'].strip(),
                        invalid_object_id=raw['invalid_object_id']) for (qid,row),raw in zip(rows.items(),raw_banks[host][task][mode])}
            else:
                for mode in MODES:
                    scored[task][host+'.'+mode]={str(row['qid']):dict(deps['diag']._selection(row,
                        prepared[task]['direct'][str(row['qid'])],end_sym)) for row in raw_banks[host][task][mode]}
                    for row in scored[task][host+'.'+mode].values():row['correct']=int(row['outcome']=='correct')
    return scored


def _change(before,after):
    n=len(before)
    gains=sum(not before[q]['correct'] and after[q]['correct'] for q in before)
    losses=sum(before[q]['correct'] and not after[q]['correct'] for q in before)
    result=dict(questions=n,gains=gains,losses=losses,delta=(gains-losses)/n)
    if 'native_id_only' in next(iter(before.values())):
        result['losses_with_new_identifier_only']=sum(before[q]['correct'] and not after[q]['correct']
            and after[q]['native_id_only'] for q in before)
    return result


def _summary(task,prepared,scored,deps):
    n=len(prepared['qids']);rate=deps['diag']._rate
    cells={}
    for cell,rows in scored.items():
        item=dict(questions=n,accuracy=rate(sum(row['correct'] for row in rows.values()),n))
        if task=='sqa3d':
            item.update({field:rate(sum(row[field] for row in rows.values()),n)
                         for field in ('native_id_only','blank','invalid_object_id')})
        else:
            outcomes=Counter(row['outcome'] for row in rows.values())
            item.update(outcomes={outcome:outcomes[outcome] for outcome in deps['diag'].OUTCOMES},
                visible_ID_validity=rate(n-outcomes['invalid'],n),candidate_compliance=rate(outcomes['correct']+outcomes['wrong_candidate'],n))
            winners=prepared['direct'];defined=[qid for qid in rows if winners[qid]['packet_winners'][cell[-1]] is not None]
            item['packet_agreement']=rate(sum(rows[qid]['source']==winners[qid]['packet_winners'][cell[-1]] for qid in defined),len(defined))
            item['packet_winner_undefined_questions']=n-len(defined)
        cells[cell]=item
    vectors=[dict(qid=qid,scene_id=prepared['rows'][qid]['scene_id'],scores={cell:rows[qid]['correct'] for cell,rows in scored.items()})
             for qid in prepared['qids']]
    primary=deps['stats'].bootstrap_interactions(vectors,WEIGHTS,replicates=REPLICATES,seed=SEED,family_size=FAMILY_SIZE)
    primary['interpretation']='One prespecified global nine-contrast family across all three tasks; scene-family uncertainty only.'
    result=dict(questions=n,scene_families=len({row['scene_family'] for row in prepared['rows'].values()}),
        primary_metric=deps['normalizer'].METRIC if task=='sqa3d' else 'strict_raw_and_mapped_native_ID',cells=cells,primary=primary,
        within_checkpoint={host:{mode+'_minus_A':_change(scored[host+'.A'],scored[host+'.'+mode]) for mode in 'TCS'} for host in HOSTS},
        across_checkpoints={mode:{before+'->'+after:_change(scored[before+'.'+mode],scored[after+'.'+mode])
            for before,after in (('parent','Atrain'),('parent','Ttrain'),('Atrain','Ttrain'))} for mode in MODES})
    if task!='sqa3d':
        direct=prepared['direct'];rankings=Counter(row['s_ranking'] for row in direct.values())
        flipped=[qid for qid in direct if direct[qid]['s_ranking']=='flipped']
        result['input_S_ranking_counts']={name:rankings[name] for name in ('flipped','preserved','tied')}
        result['joint_T_correct_S_packet_following_on_flips']={host:rate(sum(scored[host+'.T'][qid]['correct']
            and scored[host+'.S'][qid]['source']==direct[qid]['packet_winners']['S'] for qid in flipped),len(flipped)) for host in HOSTS}
        result['S_agreement_by_input_ranking']={host:{name:rate(sum(scored[host+'.S'][qid]['source']==direct[qid]['packet_winners']['S']
            for qid in direct if direct[qid]['s_ranking']==name),sum(row['s_ranking']==name for row in direct.values()))
            if name!='tied' else rate(0,0) for name in ('flipped','preserved','tied')} for host in HOSTS}
        result['transitions']={}
        for mode in MODES:
            names=[host+'.'+mode for host in HOSTS];raw=deps['diag']._transitions(scored,names,direct)
            result['transitions'][mode]={before+'->'+after:dict(questions=n,counts={old:{new:raw[before+'.'+mode+'->'+after+'.'+mode]
                ['outcome_counts'].get(old+'->'+new,0) for new in deps['diag'].OUTCOMES} for old in deps['diag'].OUTCOMES})
                for before,after in (('parent','Atrain'),('parent','Ttrain'),('Atrain','Ttrain'))}
    return result


def analyze(metadata,references,raw_banks,input_bindings,model_provenance,*,end_sym,expected_counts=None,expected_families=None):
    prepared=validate(metadata,references,raw_banks,input_bindings,model_provenance,end_sym=end_sym,
        expected_counts=expected_counts,expected_families=expected_families)
    scored=score_cells(raw_banks,references,prepared,end_sym=end_sym);deps=dependencies()
    return dict(schema_version='scalestitch_matched_exposure_CPU/v1',status='complete_declared_matrix',
        analysis_protocol_sha256=PROTOCOL_SHA,primary_global_family_size=9,replicates=REPLICATES,seed=SEED,
        accuracy_cells=36,outputs=sum(len(row['qids']) for row in prepared.values())*12,
        source_sha256=PINS,model_provenance=model_provenance,
        tasks={task:_summary(task,prepared[task],scored[task],deps) for task in TASKS},
        raw_predictions_exported=False,row_banks_exported=False,gpu_used=False,
        interpretation='Fixed within-run packet-exposure contrasts; no training-seed or cross-host uncertainty.')


def _file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,'duplicate JSON field: '+key);result[key]=value
        return result
    def invalid(value):raise ValueError('nonfinite JSON value: '+value)
    return json.loads(Path(path).read_bytes(),object_pairs_hook=unique,parse_constant=invalid)


def load_config(path):
    """Resolve JSON files from the config; file data is never treated as paths."""
    path=Path(path);config=_read_json(path)
    roles=('metadata','references','raw_banks','input_bindings','model_provenance')
    required=set(roles)|{'end_sym'};optional={'expected_counts','expected_families'}
    require(isinstance(config,dict) and required<=set(config)<=required|optional,'unknown or missing analysis configuration')
    sources={}
    def load(node,label):
        if isinstance(node,str) and node:
            source=(path.parent/node).resolve()
            digest=_file_sha(source);value=_read_json(source)
            require(_file_sha(source)==digest,'input file changed while reading: '+label)
            sources[label]=dict(path=source,sha256=digest)
            return value
        require(isinstance(node,dict) and node,'input role requires a JSON filename or nested filename map: '+label)
        return {key:load(value,label+'.'+key) for key,value in node.items()}
    loaded={role:load(config[role],role) for role in roles}
    loaded['end_sym']=config['end_sym']
    for key in optional:
        if key in config:loaded[key]=config[key]
    return loaded,sources


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.config is None:
        require(args.output is None,'--output requires --config')
        print(json.dumps(dict(status='prepared_not_executed',analysis_protocol_sha256=PROTOCOL_SHA,
            counts=COUNTS,scene_families=FAMILIES,checkpoints=HOSTS,conditions=MODES,outputs=sum(COUNTS.values())*12,
            primary_global_family_size=FAMILY_SIZE,replicates=REPLICATES,seed=SEED,gpu_used=False)))
        return 0
    require(args.output is not None and not args.output.exists(),'a new --output JSON file is required')
    config_sha=_file_sha(args.config);analyzer_sha=_file_sha(__file__)
    loaded,sources=load_config(args.config)
    result=analyze(**loaded)
    require(_file_sha(args.config)==config_sha and _file_sha(__file__)==analyzer_sha
        and all(_file_sha(item['path'])==item['sha256'] for item in sources.values()),'analysis source or input file changed during analysis')
    require(_source_fingerprints(Path(__file__).resolve().parent)==result['source_sha256'],'CPU dependency source changed during analysis')
    result['file_provenance']=dict(config_sha256=config_sha,inputs_sha256={name:item['sha256'] for name,item in sources.items()},
        analyzer_sha256=analyzer_sha,dependency_canonical_lf_sha256=dict(DEPENDENCY_CANONICAL_SHA256),
        dependency_hash_normalization='CRLF to LF; no other byte normalization',reference_analysis_sha256=REFERENCE_ANALYSIS_SHA256)
    payload=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:stream.write(payload)
    print(json.dumps(dict(status=result['status'],outputs=result['outputs'],accuracy_cells=result['accuracy_cells'],gpu_used=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
