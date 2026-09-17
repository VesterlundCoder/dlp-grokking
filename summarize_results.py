import json, os, glob

base = '/scratch/project_465002952/dlp_grokking/results/scaling'
models = ['M07','M08','M09','M10','M11','M12']
seeds = [42, 123, 456]

for model in models:
    print(f'\n=== {model} ===')
    for seed in seeds:
        d = f'{base}/{model}_s{seed}'
        spath = f'{d}/summary.json'
        mpath = f'{d}/metrics.jsonl'
        if not os.path.exists(spath):
            # Check if metrics exist (still running)
            if os.path.exists(mpath):
                lines = open(mpath).readlines()
                if lines:
                    last = json.loads(lines[-1])
                    print(f'  s{seed}: RUNNING (ep {last.get("epoch",0)}, train={last.get("train_acc",0):.4f} test={last.get("test_acc",0):.4f})')
                else:
                    print(f'  s{seed}: RUNNING (no metrics yet)')
            else:
                print(f'  s{seed}: NOT STARTED')
            continue
        s = json.load(open(spath))
        state = s.get('state','?')
        best = s.get('best_test_acc', 0)
        t50 = s.get('T_50')
        t95 = s.get('T_95')
        final_wd = s.get('final_wd', s.get('wd_schedule',{}).get('max',0))
        train_frac = s.get('train_frac', 0)
        train_size = s.get('train_size', 0)
        # Read key metrics from file
        lines = open(mpath).readlines()
        metrics = [json.loads(l) for l in lines]
        n = len(metrics)
        # Find best test acc epoch
        best_ep = 0
        best_val = 0
        for m in metrics:
            ta = m.get('test_acc', 0)
            if ta and ta > best_val:
                best_val = ta
                best_ep = m.get('epoch', 0)
        # Last entry
        last = metrics[-1] if metrics else {}
        last_ep = last.get('epoch', 0)
        last_train = last.get('train_acc', 0)
        last_test = last.get('test_acc', 0)
        last_loss = last.get('train_loss', 0)
        # WD ramps
        ramps = []
        prev_wd = 0
        for m in metrics:
            wd = m.get('weight_decay', 0)
            if wd > prev_wd + 0.001:
                ramps.append((m.get('epoch',0), wd))
                prev_wd = wd
        ramp_str = ', '.join(f'ep{e}:{w:.2f}' for e,w in ramps) if ramps else 'none'
        print(f'  s{seed}: state={state} best={best:.4f}@ep{best_ep} | last ep{last_ep} train={last_train:.4f} test={last_test:.4f} loss={last_loss:.2f} | wd_ramps={ramp_str} | train_frac={train_frac} train_size={train_size}')
