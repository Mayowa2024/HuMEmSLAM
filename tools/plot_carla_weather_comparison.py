#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load_poses(path):
    a=np.loadtxt(path).reshape(-1,3,4); out=np.repeat(np.eye(4)[None],len(a),axis=0); out[:,:3,:4]=a; return out
def frames_for_trajectory(run):
    rows=[r for r in csv.DictReader(open(run/'orb_latency.csv')) if r['event']=='track' and r['tracking_state_name']!='NO_IMAGES_YET']
    return np.array([int(r['frame_id']) for r in rows]), rows
def gt_positions(path):
    out={}
    for r in csv.DictReader(open(path)):


        out[int(r['image_index'])]=np.array([float(r['y']),-float(r['z']),float(r['x'])])
    return out
def rigid(src,dst):
    sm,dm=src.mean(0),dst.mean(0); u,_,vt=np.linalg.svd((src-sm).T@(dst-dm)); R=vt.T@u.T
    if np.linalg.det(R)<0: vt[-1]*=-1; R=vt.T@u.T
    return R,dm-R@sm
def ma(x,n=25):
    p=np.pad(x,(n//2,n-1-n//2),mode='edge'); return np.convolve(p,np.ones(n)/n,'valid')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--gt',type=Path,required=True); a=ap.parse_args()
    specs={'orbslam3_baseline_final':('ORB-SLAM3','#d62728'),'humanslam_final':('HuMemSLAM','#9467bd')}; gtmap=gt_positions(a.gt)
    plots=a.root/'plots'; metrics=a.root/'metrics'; plots.mkdir(exist_ok=True); metrics.mkdir(exist_ok=True)
    data={}; summary={}
    for key,(label,color) in specs.items():
        run=a.root/'runs'/key; est=load_poses(run/'trajectory_kitti.txt'); ids,rows=frames_for_trajectory(run); n=min(len(est),len(ids)); est,ids=est[:n],ids[:n]
        keep=np.array([i in gtmap for i in ids]); est,ids=est[keep],ids[keep]; gt=np.array([gtmap[i] for i in ids]); xyz=est[:,:3,3]
        prefix=min(100,len(xyz)); R0,t0=rigid(xyz[:prefix],gt[:prefix]); anchored=(R0@xyz.T).T+t0; err=np.linalg.norm(anchored-gt,axis=1)
        Ra,ta=rigid(xyz,gt); aligned=(Ra@xyz.T).T+ta; ape=np.linalg.norm(aligned-gt,axis=1)
        states={}; maps=set()
        for r in rows: states[r['tracking_state_name']]=states.get(r['tracking_state_name'],0)+1; maps.add(int(r['map_id']))
        data[key]=(ids,gt,aligned,err,ape,label,color)
        summary[key]={'processed_tracks':len(rows),'trajectory_poses_evaluated':len(ids),'states':states,'map_count':len(maps),'map_ids':sorted(maps),'prefix_aligned_error_rmse_m':float(np.sqrt(np.mean(err**2))),'ape_se3_rmse_m':float(np.sqrt(np.mean(ape**2)))}
    json.dump(summary,open(metrics/'summary.json','w'),indent=2)


    all_gt=[]; first=gtmap[min(gtmap)]
    for frame in range(max(gtmap)+1):
        p=gtmap.get(frame,first); T=np.eye(4); T[:3,3]=p; all_gt.append(T[:3,:4].reshape(-1))
    np.savetxt(metrics/'groundtruth_visualisation_kitti.txt',np.asarray(all_gt),fmt='%.9f')
    with open(metrics/'error_vs_frame.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['run','frame','prefix_aligned_translation_error_m'])
        for key,(ids,_,_,err,_,_,_) in data.items(): w.writerows((key,int(i),float(e)) for i,e in zip(ids,err))
    spans=[(0,1200,'ClearNoon','#fff4b2'),(1200,2400,'HardRainNoon','#b4c7e7'),(2400,3600,'ClearSunset','#f4b183'),(3600,4800,'ClearNight','#9e9e9e'),(4800,6010,'RainNight','#5b6b8c')]
    fig,ax=plt.subplots(figsize=(13,5.5))
    for lo,hi,name,c in spans: ax.axvspan(lo,hi,color=c,alpha=.18); ax.text((lo+hi)/2,.98,name,ha='center',va='top',transform=ax.get_xaxis_transform(),fontsize=8)
    for ids,_,_,err,_,label,color in data.values(): ax.plot(ids,ma(err),label=label+' (25-frame mean)',color=color,lw=2)
    ax.set(xlabel='Dataset frame',ylabel='Translation error after first-100-frame alignment (m)',title='CARLA dynamic weather: error versus frame'); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(plots/'error_vs_frame_weather.png',dpi=180); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5.5))
    for ax,(ids,gt,aligned,_,_,label,color) in zip(axs,data.values()): ax.plot(gt[:,0],gt[:,2],c='black',lw=1.3,label='Ground truth'); ax.plot(aligned[:,0],aligned[:,2],c=color,lw=1,label=label); ax.set_title(label); ax.set_aspect('equal',adjustable='datalim'); ax.grid(alpha=.2); ax.legend(); ax.set(xlabel='x (m)',ylabel='z (m)')
    fig.tight_layout(); fig.savefig(plots/'trajectories.png',dpi=180); plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
