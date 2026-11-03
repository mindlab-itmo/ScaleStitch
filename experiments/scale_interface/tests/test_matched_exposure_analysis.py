"""SYNTHETIC two-family fixtures for portable matched-exposure CPU analysis."""
import copy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import shutil

from experiments.scale_interface import matched_exposure_analysis as analysis


class MatchedExposureTests(unittest.TestCase):
    def module(self):
        return analysis

    def fixture(self):
        m=self.module();metadata={};references={};banks={host:{} for host in m.HOSTS};bindings=copy.deepcopy(banks)
        scenes=('scene0001_00','scene0001_01','scene0002_00')
        for task in m.TASKS:
            metadata[task]=[];references[task]=[]
            for index,scene in enumerate(scenes):
                values=([10,20,30],[10,20,30],[10,20,20])[index]
                pair,target=(([0,1],1),([1,2],2),([0,2],2))[index]
                objects=[dict(visible_id=v,source_row=s) for v,s in enumerate([2,0,1])]
                packets={'A':None}
                for mode in 'TCS':
                    packets[mode]=[dict(obj,value_source_row=None if mode=='C' else obj['source_row'] if mode=='T'
                        else (obj['source_row']-1)%3,value_cm=100 if mode=='C' else values[obj['source_row'] if mode=='T'
                        else (obj['source_row']-1)%3]) for obj in objects]
                base=dict(qid=str(index),scene_id=scene,scene_family=scene.split('_')[0],objects=objects,packets=packets,
                    input_fingerprint=hashlib.sha256((task+str(index)).encode()).hexdigest(),data_sha256='d'*64,
                    packet_sha256={mode:None if mode=='A' else hashlib.sha256(json.dumps(packets[mode],sort_keys=True).encode()).hexdigest()
                                   for mode in 'ATCS'},
                    prefix_budgets={mode:dict(native_tokens=10,packet_tokens=0 if mode=='A' else 8,
                        prefix_tokens=10 if mode=='A' else 18,numerical_blocks=0 if mode=='A' else 3) for mode in 'ATCS'})
                if task!='sqa3d':base.update(candidate_source_rows=pair,target_source_row=target,direction='max')
                metadata[task].append(base)
                answers=[f'<OBJ{target:03d}>'] if task!='sqa3d' else [['two'],['seat','chair'],['blue']][index]
                references[task].append(dict(question_id=str(index),scene_id=scene,answers=answers))
            for host in m.HOSTS:
                banks[host][task]={mode:[] for mode in 'ATCS'};bindings[host][task]={mode:[] for mode in 'ATCS'}
                for mode in 'ATCS':
                    for item in metadata[task]:
                        index=int(item['qid']);inverse={obj['source_row']:obj['visible_id'] for obj in item['objects']}
                        source=item.get('target_source_row')
                        text=['2','chair','blue'][index] if task=='sqa3d' else f'<OBJ{inverse[source]:03d}>'
                        if host=='parent' and mode=='A':
                            if task=='sqa3d' and index==2:text='<OBJ002>'
                            if task!='sqa3d':text=(f'<OBJ{inverse[0]:03d}>',f'<OBJ{inverse[0]:03d}>','prose answer')[index]
                        if host=='Atrain' and mode=='T' and index==0:text='one' if task=='sqa3d' else f'<OBJ{inverse[0]:03d}>'
                        if host=='Ttrain' and mode=='S' and index==0:text='one' if task=='sqa3d' else f'<OBJ{inverse[0]:03d}>'
                        if task=='sqa3d' and host=='parent' and mode=='C' and index==2:text=''
                        mapped=text
                        for visible,original in [(obj['visible_id'],obj['source_row']) for obj in item['objects']]:
                            mapped=mapped.replace(f'<OBJ{visible:03d}>',f'@SOURCE{original}@')
                        for original in range(3):mapped=mapped.replace(f'@SOURCE{original}@',f'<OBJ{original:03d}>')
                        banks[host][task][mode].append(dict(qid=item['qid'],scene_id=item['scene_id'],condition=mode,
                            raw_prediction=text+'</s>',prediction=mapped,invalid_object_id=False,correct=1,
                            effective_generation=dict(max_new_tokens=32,num_beams=1,do_sample=False,
                                use_cache=True,repetition_penalty=1.),**item['prefix_budgets'][mode]))
                        bindings[host][task][mode].append(dict(qid=item['qid'],scene_id=item['scene_id'],
                            input_fingerprint=item['input_fingerprint'],data_sha256=item['data_sha256'],
                            packet_sha256=item['packet_sha256'][mode],prefix_budget=item['prefix_budgets'][mode]))
        return dict(metadata=metadata,references=references,raw_banks=banks,input_bindings=bindings,
            model_provenance={host:dict(host_checkpoint_sha256=str(index)*64) for index,host in enumerate(m.HOSTS,1)},
            end_sym='</s>',expected_counts={task:3 for task in m.TASKS},expected_families={task:2 for task in m.TASKS})

    def test_describe_is_inert_and_has_full_production_scope(self):
        m=self.module()
        with redirect_stdout(io.StringIO()) as output:self.assertEqual(m.main([]),0)
        value=json.loads(output.getvalue());self.assertEqual(value['outputs'],142044)
        self.assertEqual(value['primary_global_family_size'],9);self.assertEqual(value['replicates'],100000)
        self.assertEqual(m.COUNTS,dict(radius=4161,height=4157,sqa3d=3519))

    def test_complete_36_cells_global_m9_and_raw_based_scores(self):
        m=self.module();result=m.analyze(**self.fixture())
        self.assertEqual(result['outputs'],108);self.assertEqual(result['accuracy_cells'],36)
        self.assertEqual(result['primary_global_family_size'],9)
        for task,summary in result['tasks'].items():
            self.assertEqual(summary['questions'],3)
            self.assertEqual(len(summary['cells']),12)
            for mode,effect in summary['primary']['contrasts'].items():
                self.assertAlmostEqual(effect['delta'],2/3 if mode=='T_by_S' else 1/3)
                self.assertEqual(effect['ci_bonferroni'],[0.,1.] if mode=='T_by_S' else [0.,.5])
                self.assertEqual(effect['bonferroni_confidence'],1-.05/9)
            self.assertEqual(summary['primary']['replicates'],100000)
            self.assertEqual(summary['primary']['family_size'],9)
        direct=result['tasks']['radius']
        self.assertEqual(direct['cells']['parent.A']['outcomes'],dict(correct=0,wrong_candidate=1,out_of_pair=1,invalid=1))
        transition=direct['transitions']['A']['parent->Atrain']['counts']
        self.assertEqual(transition['wrong_candidate']['correct'],1)
        self.assertEqual(transition['out_of_pair']['correct'],1);self.assertEqual(transition['invalid']['correct'],1)
        self.assertEqual(direct['joint_T_correct_S_packet_following_on_flips']['Ttrain'],dict(count=1,denominator=1,fraction=1.))
        self.assertEqual(direct['input_S_ranking_counts'],dict(flipped=1,preserved=1,tied=1))
        natural=result['tasks']['sqa3d']
        self.assertEqual(natural['primary_metric'],'host_normalized_em1_3dgraphllm_19d65cd9_v1')
        self.assertEqual(natural['cells']['parent.A']['native_id_only']['count'],1)
        self.assertEqual(natural['within_checkpoint']['Atrain']['T_minus_A']['losses'],1)
        self.assertNotIn('input_S_ranking_counts',natural)

    def test_all_matrix_and_input_bindings_checked_before_any_score(self):
        m=self.module()
        for kind in ('missing','duplicate','order','scene','input','model_in_binding','donor','reference','family'):
            data=self.fixture()
            if kind=='missing':data['raw_banks']['Ttrain']['sqa3d']['S'].pop()
            if kind=='duplicate':data['raw_banks']['Ttrain']['sqa3d']['S'][-1]=data['raw_banks']['Ttrain']['sqa3d']['S'][0]
            if kind=='order':data['input_bindings']['Atrain']['height']['T'].reverse()
            if kind=='scene':data['raw_banks']['Ttrain']['sqa3d']['S'][0]['scene_id']='scene9999_00'
            if kind=='input':data['input_bindings']['Ttrain']['height']['T'][0]['input_fingerprint']='e'*64
            if kind=='model_in_binding':data['input_bindings']['Ttrain']['height']['T'][0]['host_checkpoint_sha256']='f'*64
            if kind=='donor':data['metadata']['height'][0]['packets']['S'][0]['value_source_row']=2
            if kind=='reference':data['references']['radius'][0]['answers']=['<OBJ000>']
            if kind=='family':data['metadata']['sqa3d'][0]['scene_family']='scene9999'
            deps=m.dependencies()
            with self.subTest(kind=kind),patch.object(deps['scorer'],'score_predictions',side_effect=AssertionError('scored too early')):
                with self.assertRaises(ValueError):m.analyze(**data)

    def test_mapping_is_checked_and_prose_remains_a_full_denominator_error(self):
        m=self.module();data=self.fixture()
        data['raw_banks']['parent']['radius']['T'][0]['prediction']='<OBJ000>'
        with self.assertRaisesRegex(ValueError,'mapping|raw'):m.validate(**data)
        data=self.fixture();prepared=m.validate(**data);scored=m.score_cells(data['raw_banks'],data['references'],prepared,end_sym='</s>')
        self.assertEqual(scored['radius']['parent.A']['2']['outcome'],'invalid')
        self.assertEqual(scored['sqa3d']['parent.C']['2']['correct'],0)
        self.assertEqual(scored['sqa3d']['Ttrain.T']['0']['correct'],1)
        self.assertEqual(scored['sqa3d']['parent.A']['2']['native_id_only'],1)

    def test_effective_generation_and_prefix_traces_are_verified_before_scoring(self):
        m=self.module();data=self.fixture()
        m.validate(**data)
        data['raw_banks']['Ttrain']['radius']['T'][0]['effective_generation']['decision_mode']='free'
        m.validate(**data)
        mutations={
            'max_new_tokens':31,'num_beams':2,'do_sample':True,'use_cache':False,'repetition_penalty':3.,
            'decision_mode':'candidate_logits','missing_generation':None,'missing_num_beams':None,
            'native_tokens':9999,'packet_tokens':9999,'prefix_tokens':9999,'numerical_blocks':9999}
        for name,value in mutations.items():
            data=self.fixture();row=data['raw_banks']['Ttrain']['radius']['T'][0]
            if name=='missing_generation':row.pop('effective_generation')
            elif name=='missing_num_beams':row['effective_generation'].pop('num_beams')
            elif name in ('native_tokens','packet_tokens','prefix_tokens','numerical_blocks'):row[name]=value
            else:row['effective_generation'][name]=value
            with self.subTest(field=name),self.assertRaisesRegex(ValueError,'decoder|generation|prefix'):
                m.validate(**data)

    def config(self,root):
        data=self.fixture();config={key:data[key] for key in ('end_sym','expected_counts','expected_families')}
        for role in ('metadata','references','raw_banks','input_bindings','model_provenance'):
            path=root/(role+'.json');path.write_text(json.dumps(data[role]),encoding='utf-8');config[role]=path.name
        path=root/'analysis.json';path.write_text(json.dumps(config),encoding='utf-8')
        return path,root/'result.json'

    def test_canonical_dependency_whitelist_accepts_lf_and_crlf_but_records_actual_bytes(self):
        base=Path(analysis.__file__).parent
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);lf=root/'lf';crlf=root/'crlf';lf.mkdir();crlf.mkdir()
            for name in ('text_baseline_statistics','reviewer_diagnostics','host_normalization','prediction_scoring','natural_qa_metrics'):
                canonical=(base/(name+'.py')).read_bytes().replace(b'\r\n',b'\n')
                (lf/(name+'.py')).write_bytes(canonical);(crlf/(name+'.py')).write_bytes(canonical.replace(b'\n',b'\r\n'))
            first=analysis._source_fingerprints(lf);second=analysis._source_fingerprints(crlf)
            self.assertNotEqual(first['text_baseline_statistics'],second['text_baseline_statistics'])
            self.assertEqual(first['text_baseline_statistics'],hashlib.sha256((lf/'text_baseline_statistics.py').read_bytes()).hexdigest())
            with (lf/'text_baseline_statistics.py').open('ab') as stream:stream.write(b'\n# changed source\n')
            with self.assertRaisesRegex(ValueError,'source'):analysis._source_fingerprints(lf)

    def test_dependency_provenance_does_not_mutate_prior_results(self):
        analysis.dependencies();prior=analysis.PINS;expected=dict(prior)
        analysis.dependencies()
        self.assertIsNot(analysis.PINS,prior);self.assertEqual(prior,expected)
        for name,digest in prior.items():
            self.assertEqual(digest,hashlib.sha256((Path(analysis.__file__).parent/(name+'.py')).read_bytes()).hexdigest())

    def test_portable_cli_binds_config_all_inputs_and_actual_dependency_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            config,output=self.config(Path(folder))
            with redirect_stdout(io.StringIO()):self.assertEqual(analysis.main(['--config',str(config),'--output',str(output)]),0)
            result=json.loads(output.read_text());self.assertEqual(result['outputs'],108)
            self.assertEqual(result['accuracy_cells'],36)
            provenance=result['file_provenance']
            self.assertEqual(provenance['config_sha256'],hashlib.sha256(config.read_bytes()).hexdigest())
            self.assertEqual(set(provenance['inputs_sha256']),{'metadata','references','raw_banks','input_bindings','model_provenance'})
            for role,digest in provenance['inputs_sha256'].items():self.assertEqual(digest,hashlib.sha256((config.parent/(role+'.json')).read_bytes()).hexdigest())
            self.assertEqual(result['source_sha256'],analysis._source_fingerprints(Path(analysis.__file__).parent))
            self.assertEqual(result['analysis_protocol_sha256'],'6c39b847a11828a74cf173241798b78566c4fe690b9b057f80fa74b331b617e1')
            self.assertEqual(provenance['dependency_hash_normalization'],'CRLF to LF; no other byte normalization')
            self.assertFalse(result['raw_predictions_exported']);self.assertFalse(result['gpu_used'])
            with self.assertRaises(ValueError):analysis.main(['--config',str(config),'--output',str(output)])

    def test_nested_file_config_and_changed_input_are_checked_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config,output=self.config(root);value=json.loads(config.read_text())
            metadata=json.loads((root/'metadata.json').read_text());value['metadata']={}
            for task,rows in metadata.items():
                filename='metadata_'+task+'.json';(root/filename).write_text(json.dumps(rows));value['metadata'][task]=filename
            config.write_text(json.dumps(value))
            loaded,sources=analysis.load_config(config)
            self.assertEqual(loaded['metadata'],metadata)
            self.assertEqual({key for key in sources if key.startswith('metadata.')},{'metadata.radius','metadata.height','metadata.sqa3d'})
            def changed(**kwargs):
                (root/'references.json').write_text('[]')
                return dict(source_sha256=analysis._source_fingerprints(Path(analysis.__file__).parent))
            with patch.object(analysis,'analyze',side_effect=changed),self.assertRaisesRegex(ValueError,'changed'):
                analysis.main(['--config',str(config),'--output',str(output)])
            self.assertFalse(output.exists())

    def test_default_and_help_do_not_load_dependencies_or_create_outputs(self):
        with patch.object(analysis,'dependencies',side_effect=AssertionError('inert CLI loaded dependencies')):
            with redirect_stdout(io.StringIO()):self.assertEqual(analysis.main([]),0)
            with redirect_stdout(io.StringIO()),self.assertRaises(SystemExit) as stopped:analysis.main(['--help'])
            self.assertEqual(stopped.exception.code,0)


if __name__=='__main__':unittest.main()
