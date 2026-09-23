import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import sys,json,argparse,hashlib
from pathlib import Path
import numpy as np,torch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'));torch.set_num_threads(2)
parser=argparse.ArgumentParser(description='Historical model-state probes; requires external expression inputs and archived final checkpoints.')
parser.add_argument('--input-dir',type=Path,default=R/'inputs')
parser.add_argument('--checkpoint-dir',type=Path,default=R/'training',help='Root containing each arm directory and its last.pt; defaults to the archived states.')
parser.add_argument('--output-dir',type=Path,default=R.parents[1]/'reproduced'/'model-probes')
args=parser.parse_args()
manifest=json.loads((R/'input_manifest.json').read_text())
for dataset in ('setty','dentate'):
 path=args.input_dir/f'{dataset}.npz'
 if not path.is_file():
  parser.error(f'{path.name} is not distributed. Supply lawfully obtained, hash-matching expression snapshots with --input-dir; see SOURCE_DATA.md.')
 if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['inputs/'+path.name]:
  parser.error(f'Source-input hash mismatch: {path.name}')
if args.output_dir.resolve().is_relative_to(R):
 parser.error('Choose an output directory outside archived evidence/reproduction.')
checkpoint_hashes={}
for dataset in ('setty','dentate'):
 for seed in range(3):
  for alpha in (.1,1.):
   name=f'{dataset}_s{seed}_a{alpha:g}'
   checkpoint=args.checkpoint_dir/name/'last.pt'
   if not checkpoint.is_file():
    parser.error(f'Missing state {name}/last.pt. Extract this study\'s checkpoint archive or select the intended model-state root with --checkpoint-dir.')
   checkpoint_hashes[name]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
from models.topic_transformer import TopicODETransformerModel
rows=[];out=args.output_dir/'probes';out.mkdir(parents=True,exist_ok=True)
(args.output_dir/'checkpoint-identities.json').write_text(json.dumps(checkpoint_hashes,indent=2)+'\n')
for ds in ['setty','dentate']:
 d=np.load(args.input_dir/f'{ds}.npz');x=torch.tensor(d['test_raw'][:64],dtype=torch.float32);xn=torch.tensor(d['test_norm'][:64],dtype=torch.float32)
 for seed in range(3):
  for alpha in [.1,1.]:
   name=f'{ds}_s{seed}_a{alpha:g}';m=TopicODETransformerModel(input_dim=x.shape[1],n_topics=10,dropout=0.,cell_topic_prior=torch.full((10,),alpha));m.load_state_dict(torch.load(args.checkpoint_dir/name/'last.pt',map_location='cpu',weights_only=True))
   for mode in ['eval','train']:
    m.train(mode=='train');saved={}
    with torch.no_grad():
     base=m.encode(x,x_norm=xn);saved['ordinary']=base.numpy()
     for intervention in ['reverse_tokens','mean_tokens','duplicate_first']:
      def hook(module,args):
       t=args[0]
       if intervention=='reverse_tokens':t=t.flip(1)
       elif intervention=='mean_tokens':t=t.mean(1,keepdim=True).expand_as(t)
       else:t=t[:,:1,:].expand_as(t)
       return (t,)
      h=m.ae.encoder.transformer.register_forward_pre_hook(hook);z=m.encode(x,x_norm=xn);h.remove();saved[intervention]=z.numpy()
      rows.append(dict(dataset=ds,seed=seed,alpha=alpha,mode=mode,intervention=intervention,max_abs=float((z-base).abs().max()),mean_l1=float((z-base).abs().sum(1).mean())))
     # First 16 focal cells: alter only the other 48 cells, and compare isolated calls.
     altered=xn.clone();altered[16:]*=10
     variants={'alter_companions':m.encode(x,x_norm=altered)[:16],'reverse_batch':m.encode(x.flip(0),x_norm=xn.flip(0)).flip(0)[:16],'isolated':torch.cat([m.encode(x[i:i+1],x_norm=xn[i:i+1]) for i in range(16)])}
     for intervention,z in variants.items():
      saved[intervention]=z.numpy();rows.append(dict(dataset=ds,seed=seed,alpha=alpha,mode=mode,intervention=intervention,max_abs=float((z-base[:16]).abs().max()),mean_l1=float((z-base[:16]).abs().sum(1).mean())))
    np.savez_compressed(out/f'{name}_{mode}.npz',**saved)
(args.output_dir/'probe_results.json').write_text(json.dumps(rows,indent=2))
for name in ['reverse_tokens','mean_tokens','duplicate_first','alter_companions','reverse_batch','isolated']:
 r=[a for a in rows if a['intervention']==name];print(name,'max_abs',max(a['max_abs'] for a in r),'mean_L1',np.mean([a['mean_l1'] for a in r]))
