import json, sys

for model in ['M07', 'M08']:
    for seed in [42, 123, 456]:
        path = f'/scratch/project_465002952/dlp_grokking/results/scaling/{model}_s{seed}/metrics.jsonl'
        try:
            lines = open(path).readlines()
            metrics = [json.loads(l) for l in lines]
            print(f'=== {model} seed={seed} ({len(metrics)} entries) ===')
            prev_te = 0.0
            for m in metrics:
                ta = m.get('train_acc')
                te = m.get('test_acc')
                ep = m.get('epoch', 0)
                tl = m.get('train_loss', 0)
                wd = m.get('weight_decay', 0)
                wn = m.get('weight_norm', 0)
                if ta is not None and te is not None:
                    if abs(te - prev_te) > 0.1 or ep < 100 or ep % 20000 == 0 or ep == metrics[-1].get('epoch', 0):
                        print(f'  Ep {ep:6d} | train={ta:.4f} test={te:.4f} loss={tl:.4f} wd={wd:.4f} wnorm={wn:.1f}')
                    prev_te = te
            spath = f'/scratch/project_465002952/dlp_grokking/results/scaling/{model}_s{seed}/summary.json'
            try:
                s = json.load(open(spath))
                print(f'  SUMMARY: state={s.get("state")} best={s.get("best_test_acc",0):.4f} T_mem={s.get("T_mem")} T_50={s.get("T_50")} T_95={s.get("T_95")} final_wd={s.get("final_wd",0):.4f}')
            except:
                pass
            print()
        except Exception as e:
            print(f'{model} seed={seed}: {e}')
            print()
