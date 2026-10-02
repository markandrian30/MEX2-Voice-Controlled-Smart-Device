"""Direct-file holdout evaluation; never opens a microphone or executes commands."""
import argparse, csv, hashlib, io, json, sys, time, urllib.request
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from runtime import load_model, load_wav, wav_to_logmel, accept, intent, SR

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio-dir',type=Path,help='Existing directory containing audio/0000.wav etc.')
    p.add_argument('--parquet',type=Path,help='Existing exact holdout parquet; otherwise download and checksum it')
    p.add_argument('--min-confidence',type=float,default=.25)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if not 0 <= a.min_confidence <= 1: p.error('Threshold must be in [0,1]')
    manifest=json.loads((ROOT/'data/holdout_manifest.json').read_text())
    audio_dir=a.audio_dir or ROOT/'.cache/holdout'
    if not a.audio_dir:
        import pyarrow.parquet as pq
        source=json.loads((ROOT/'data/holdout_source.json').read_text())
        parquet=a.parquet or ROOT/'.cache/holdout.parquet'
        if not parquet.exists():
            parquet.parent.mkdir(parents=True,exist_ok=True)
            urllib.request.urlretrieve(source['url'],parquet)
        if hashlib.sha256(parquet.read_bytes()).hexdigest()!=source['parquet_sha256']:
            raise ValueError('Holdout version differs. Supply the original parquet with --parquet; do not mix dataset versions.')
        source_rows={r['file']:r for r in pq.read_table(parquet).to_pylist()}
        for r in manifest:
            raw=source_rows[r['source_file']]['audio']['bytes']
            if hashlib.sha256(raw).hexdigest()!=r['sha256']: raise ValueError('Audio checksum mismatch')
            path=audio_dir/r['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    model,cp=load_model(ROOT/'model/best_model.pt')
    out=a.output or ROOT/'evaluation_runs'/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    with torch.inference_mode():
        for i,r in enumerate(manifest):
            path=audio_dir/r['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']: raise ValueError('Audio checksum mismatch')
            wav=load_wav(path)
            if i==0:
                x=wav_to_logmel(wav).unsqueeze(0)
                for _ in range(10):model(x)
            times=[]
            for _ in range(3):
                t=time.perf_counter();prob=model(wav_to_logmel(wav).unsqueeze(0)).softmax(1)[0]
                times.append((time.perf_counter()-t)*1000)
            raw=cp['labels'][int(prob.argmax())];confidence=float(prob.max());pred=accept(raw,confidence,a.min_confidence)
            records.append(dict(**r,raw_prediction=raw,prediction=pred,confidence=confidence,pred_intent=intent(pred),duration_s=len(wav)/SR,pipeline_ms=float(np.median(times))))
            if (i+1)%25==0:print(f'{i+1}/{len(manifest)}',flush=True)
    def summarize(rows):
        negatives=[r for r in rows if r['expected_label']=='OUT_OF_SCOPE']
        positives=[r for r in rows if r['expected_label']!='OUT_OF_SCOPE']
        return dict(recordings=len(rows),command_accuracy=sum(r['prediction']==r['expected_label'] for r in rows)/len(rows),intent_accuracy=sum(r['pred_intent']==r['expected_intent'] for r in rows)/len(rows),false_accepts=sum(r['prediction']!='OUT_OF_SCOPE' for r in negatives),out_of_scope=len(negatives),false_accept_rate=sum(r['prediction']!='OUT_OF_SCOPE' for r in negatives)/len(negatives) if negatives else None,valid_rejections=sum(r['prediction']=='OUT_OF_SCOPE' for r in positives),pipeline_p95_ms=float(np.percentile([r['pipeline_ms'] for r in rows],95)),mean_rtf=float(np.mean([r['pipeline_ms']/1000/r['duration_s'] for r in rows])))
    summary=dict(min_confidence=a.min_confidence,threads=4,runtime=torch.__version__,model_sha256=hashlib.sha256((ROOT/'model/best_model.pt').read_bytes()).hexdigest(),full=summarize(records),phrase_matched_with_oos=summarize([r for r in records if r['phrase_matched'] or r['expected_label']=='OUT_OF_SCOPE']),timing='Median of 3 preprocessing + forward + softmax passes per file; 10 warmups; no microphone, VAD, wake gate or actions')
    with (out/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print('Results:',out)

if __name__=='__main__':main()
