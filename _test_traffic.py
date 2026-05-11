import re, os, statistics, pathlib
BB   = pathlib.Path('/home/yuyqin/ETH_Master_Study/Green_Routing/network-energy-efficiency-research/lan-mon2021/bb-usage-logs')
TOPO = pathlib.Path('/home/yuyqin/ETH_Master_Study/Green_Routing/network-energy-efficiency-research/switch-network-topology/switch-network-topology.txt')

def load_latest(pair_base):
    files = sorted(f for f in os.listdir(BB) if f.startswith(pair_base + '.'))
    if not files: return None
    ins, outs = [], []
    with open(BB / files[-1]) as f:
        for line in f:
            p = line.split()
            if len(p) >= 4:
                try:
                    iv, ov = float(p[1]), float(p[3])
                    if iv > 0: ins.append(iv)
                    if ov > 0: outs.append(ov)
                except: pass
    if not ins: return None
    return files[-1], statistics.mean(ins), max(ins), statistics.mean(outs), max(outs)

PAT = re.compile(r'^(\S+)\s+\S+\s+<=>\s+(\S+)\s+\S+\s+\(([^)]+)\)')
edge_cap = {}
with open(TOPO) as f:
    for line in f:
        m = PAT.match(line.strip())
        if not m: continue
        u,v,bw = m.group(1),m.group(2),float(m.group(3))
        cu = re.sub(r'\d+$','',u[3:]).lower()
        cv = re.sub(r'\d+$','',v[3:]).lower()
        key = tuple(sorted([cu,cv]))
        edge_cap[key] = max(edge_cap.get(key,0), bw)

pairs = sorted(set(f.rsplit('.',1)[0] for f in os.listdir(BB)))
print("%-15s %12s %12s %10s %9s" % ('Pair','AvgIn_Mbps','MaxIn_Mbps','Cap_Gbps','MaxUtil%'))
for pair in pairs:
    parts = pair.split('-')
    c1, c2 = parts[0].lower(), parts[1].lower()
    key = tuple(sorted([c1,c2]))
    r = load_latest(pair)
    if not r: continue
    fname, ai, mi, ao, mo = r
    cap = edge_cap.get(key)
    if cap:
        util = max(mi,mo) / (cap/1e6) * 100
        print("%-15s %12.0f %12.0f %10.1f %9.1f%%" % (pair, ai, mi, cap/1e9, util))
