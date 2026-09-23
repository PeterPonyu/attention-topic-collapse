import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import sys,json,time,hashlib,argparse
from pathlib import Path
import numpy as np,torch
from sklearn.model_selection import KFold,cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial.distance import pdist
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'));torch.set_num_threads(2)
parser=argparse.ArgumentParser(description='Historical CUDA training implementation, outside the verified CPU workflow.')
parser.add_argument('--input-dir',type=Path,default=R/'inputs')
parser.add_argument('--output-dir',type=Path,default=R.parents[1]/'reproduced'/'training')
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
from models.topic_transformer import TopicODETransformerModel

def hashstate(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2))
records=[];start=time.time();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
for ds in ['setty','dentate']:
 d=np.load(args.input_dir/f'{ds}.npz');x=torch.tensor(d['train_raw'],dtype=torch.float32,device='cuda');xn=torch.tensor(d['train_norm'],dtype=torch.float32,device='cuda')
 for seed in range(3):
  rng=np.random.default_rng(90000+seed);order=np.array([rng.permutation(len(x))[:2048].reshape(16,128) for _ in range(200)]);stream=rng.integers(0,2**31-1,size=3200)
  np.savez_compressed(out/f'{ds}_s{seed}_schedule.npz',order=order,random_seeds=stream)
  torch.manual_seed(seed);base=TopicODETransformerModel(input_dim=x.shape[1],n_topics=10,dropout=0.);ae={k:v.clone() for k,v in base.ae.state_dict().items()};del base
  for alpha in [.1,1.]:
   name=f'{ds}_s{seed}_a{alpha:g}';folder=out/name;folder.mkdir(exist_ok=True)
   if (folder/'result.json').exists():records.append(json.loads((folder/'result.json').read_text()));continue
   torch.manual_seed(seed);m=TopicODETransformerModel(input_dim=x.shape[1],n_topics=10,dropout=0.,cell_topic_prior=torch.full((10,),alpha),topic_word_prior=torch.ones(x.shape[1]));m.ae.load_state_dict(ae)
   initial_hash=hashstate(m.ae.state_dict());prior_hash=hashstate({k:v for k,v in m.state_dict().items() if k.startswith('prior_')});torch.save(m.state_dict(),folder/'initial.pt');m=m.cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
   began=time.time();step=0;losses=[]
   for epoch in range(200):
    m.train();ls=[]
    for ids in order[epoch]:
     torch.manual_seed(int(stream[step]));opt.zero_grad(set_to_none=True);o=m(x[ids],x_norm=xn[ids]);loss=m.compute_loss(o)['total_loss'];assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),10.,error_if_nonfinite=True);opt.step();ls.append(float(loss.detach()));step+=1
    losses.append(float(np.mean(ls)))
    if epoch==0 or (epoch+1)%50==0:print(name,epoch+1,round(time.time()-began,1),losses[-1],flush=True)
   torch.save({k:v.cpu() for k,v in m.state_dict().items()},folder/'last.pt');m.eval()
   with torch.no_grad():z=m.encode(torch.tensor(d['test_raw'],dtype=torch.float32,device='cuda'),x_norm=torch.tensor(d['test_norm'],dtype=torch.float32,device='cuda')).cpu().numpy()
   np.savez_compressed(folder/'latent.npz',latent=z)
   pp=np.exp(-(z*np.log(np.maximum(z,1e-30))).sum(1));span=float(np.ptp(z,axis=0).max());occ=int(len(np.unique(z.argmax(1))));js=float(np.mean(pdist(z.astype(float),metric='jensenshannon')**2));deg=js<=1e-12
   scores=None
   if ds=='setty' and not deg:scores=cross_val_score(KNeighborsRegressor(n_neighbors=15),z,d['branch'],cv=KFold(5,shuffle=True,random_state=42),scoring='r2',n_jobs=1).tolist()
   result=dict(dataset=ds,seed=seed,alpha=alpha,updates=step,presentations=step*128,epochs=200,ae_initial_hash=initial_hash,prior_hash=prior_hash,seconds=time.time()-began,mean_perplexity=float(pp.mean()),max_coordinate_range=span,argmax_topics=occ,exact_pair_js=js,degenerate=bool(deg),collapsed=bool(occ==1 and span<=1e-8),branch_r2_folds=scores,branch_r2=float(np.mean(scores)) if scores else None,loss_by_epoch=losses)
   dump(folder/'result.json',result);records.append(result);dump(out/'results.json',records);print('FINISHED',name,json.dumps({k:v for k,v in result.items() if k!='loss_by_epoch'}),flush=True);del m,opt;torch.cuda.empty_cache()
dump(out/'completion.json',dict(runs=len(records),seconds=time.time()-start));print('ALL COMPLETE',flush=True)
