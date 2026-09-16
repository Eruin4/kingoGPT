"""Fresh collection context for the KingoGPT-backed Hermes daily briefing."""
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from kingogpt.icampus import PROJECT, TZ, clean, load, write

def context(data, changes):
    if data.get('status') != 'collected' or data.get('gaps'):
        raise ValueError('정상 수집 결과가 아님')
    at=datetime.fromisoformat(data['checked_at']).astimezone(TZ)
    monday=at.date()-timedelta(days=at.weekday())
    fields=('id','course_id','kind','category','title','body','url','week','due_at','unlock_at','lock_at','site_status','locked','content_id','content_type','posted_at','updated_at','attachments')
    return clean({'checked_at':data['checked_at'],'week_start':str(monday),'week_end':str(monday+timedelta(days=6)),
      'courses':data['courses'],'items':[{k:x[k] for k in fields if k in x} for x in data['items']],
      'coverage':data['coverage'],'changes':changes,'status':'collected'})

def run(existing=False):
    root=PROJECT/'state/icampus'
    if not existing:
        before=load(root/'last-collected.json',{}).get('checked_at')
        result=subprocess.run([str(PROJECT/'.venv/bin/python'),'-m','kingogpt.icampus','--budget','900'],cwd=PROJECT,capture_output=True,text=True,timeout=930)
        if result.returncode:raise RuntimeError('오늘 iCampus 수집 실패')
    data=load(root/'last-collected.json',{})
    if not existing and data.get('checked_at')==before:raise RuntimeError('새 정상 수집 결과 없음')
    if datetime.fromisoformat(data['checked_at']).astimezone(TZ).date()!=datetime.now(TZ).date():raise RuntimeError('오늘 정상 수집 결과 없음')
    payload=context(data,load(root/'changes.json',[]))
    write(root/'briefing-input.json',payload)
    from kingogpt.icampus_hermes import compose
    print(compose(payload))
    return 0

def main():
    import argparse,fcntl
    parser=argparse.ArgumentParser()
    parser.add_argument('--existing',action='store_true',help='Verify with an existing successful collection from today')
    args=parser.parse_args()
    root=PROJECT/'state/icampus';root.mkdir(parents=True,exist_ok=True)
    with open(root/'briefing.lock','a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return run(args.existing)
        except Exception as exc:
            reason=str(exc) if isinstance(exc,(RuntimeError,ValueError)) else type(exc).__name__
            write(root/'briefing-failure.json',{'at':datetime.now(TZ).isoformat(),'reason':reason})
            print('체크리스트 생성 실패: '+reason)
            return 1

if __name__=='__main__':sys.exit(main())
