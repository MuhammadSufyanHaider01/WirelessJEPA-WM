"""Plot V1 learning curves or policy metric confidence intervals."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--csv', required=True); parser.add_argument('--output', required=True); parser.add_argument('--metric', default='return'); parser.add_argument('--learning', action='store_true'); args=parser.parse_args()
    rows=[]
    with open(args.csv, newline='') as handle:
        for row in csv.DictReader(handle):
            row[args.metric]=float(row[args.metric]); row['episode']=int(float(row.get('episode',0))); rows.append(row)
    groups=defaultdict(list)
    for row in rows: groups[row.get('policy','run')].append(row)
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); fig, ax=plt.subplots(figsize=(8,4.8))
    if args.learning:
        for name, values in groups.items():
            by_episode=defaultdict(list)
            for row in values: by_episode[row['episode']].append(row[args.metric])
            episodes=sorted(by_episode); means=np.array([np.mean(by_episode[e]) for e in episodes]); cis=np.array([1.96*np.std(by_episode[e],ddof=1)/np.sqrt(len(by_episode[e])) if len(by_episode[e])>1 else 0.0 for e in episodes])
            ax.plot(episodes, means, label=name); ax.fill_between(episodes, means-cis, means+cis, alpha=.15)
        ax.set_xlabel('Episode'); ax.set_ylabel(args.metric); ax.legend()
    else:
        names=list(groups); means=np.array([np.mean([r[args.metric] for r in groups[n]]) for n in names]); cis=np.array([1.96*np.std([r[args.metric] for r in groups[n]],ddof=1)/np.sqrt(len(groups[n])) if len(groups[n])>1 else 0.0 for n in names])
        ax.bar(names, means, yerr=cis, capsize=4); ax.set_ylabel(args.metric); ax.tick_params(axis='x', rotation=25)
    ax.grid(axis='y', alpha=.25); fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
if __name__ == '__main__': main()
