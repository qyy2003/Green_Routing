"""Link ends from SWITCH's public reverse DNS.

SWITCH names router interfaces in reverse DNS: ``swiEZ2-400GE-0-0-0-3.switch.ch`` is
``ez2/fourhundredgige0_0_0_3`` in the export. The two ends of a point-to-point link share
a /31, or a /30 whose network address is named ``NET-BB-P2P-<A>-<B>`` (IPv4), or a /64
(IPv6). That gives far ends for ports the traffic matching cannot pair, in particular
trunks into routers that are missing from the SNMP export. DNS shows today's
configuration, not the past.

Sweeps (slow, need ``dig``; snapshots are kept in the reference directory):

    sweep_v4    PTR of every address in 130.59.0.0/16 (65,536 queries)
    walk_v6     2001:620:0::/48 walked down to /64s with RFC 8020 NXDOMAIN pruning,
                then PTR of ::1 .. ::6 in each existing /64
"""
from __future__ import annotations

import concurrent.futures as cf
import ipaddress
import re
import subprocess
from pathlib import Path

import pandas as pd

V4_NET = "130.59.0.0/16"
V6_ZONE = "0.0.0.0.0.2.6.0.1.0.0.2.ip6.arpa"          # 2001:0620:0000::/48

TYPES = {"400ge": "fourhundredgige", "100ge": "hundredgige", "40ge": "fortygige",
         "25ge": "twentyfivegige", "1ge": "gigabitethernet", "ge": "gigabitethernet",
         "b": "bundle-ether", "p": "port-channel", "e": "ethernet"}


def iface(name: str) -> str | None:
    """PTR name -> ``<device>/<interface>`` as the export spells it, or None."""
    m = re.match(r"swi([a-z0-9-]+?)-((?:\d+ge|ge|b|p|e|tunnel-ip))-?([\d-]*)\.switch\.ch$", name)
    if not m:
        return None
    dev, t, path = m.groups()
    nums = [x for x in path.split("-") if x != ""]
    if t == "10ge":
        t2 = "tengige" if len(nums) >= 4 else "tengigabitethernet"
    elif t == "tunnel-ip":
        return f"{dev}/tunnel-ip{''.join(nums)}"
    else:
        t2 = TYPES.get(t, t)
    if t2 in ("bundle-ether", "port-channel"):
        return f"{dev}/{t2}{nums[0]}" + (f".{nums[1]}" if len(nums) > 1 else "")
    if t2 == "ethernet":   # Arista Ethernet29/4.2005 -> ethernet29_4 (+ .vlan)
        return f"{dev}/ethernet{'_'.join(nums[:2])}" + (f".{nums[2]}" if len(nums) > 2 else "")
    return f"{dev}/{t2}{'_'.join(nums)}"


def read_ptr(path: str | Path) -> list[tuple[str, str]]:
    """``<ip> <name>`` lines -> [(ip, lower-case name without the trailing dot)]."""
    out = []
    for line in open(path):
        f = line.split()
        if len(f) >= 2:
            out.append((f[0], f[1].rstrip(".").lower()))
    return out


def link_map(v4: list[tuple[str, str]], v6: list[tuple[str, str]] = ()) -> pd.DataFrame:
    """Every SWITCH router address with its interface and the far end's.
    Columns: ip, name, link, peer_name, peer_link, p2p_net."""
    ptr = {int(ipaddress.ip_address(ip)): name for ip, name in v4}
    recs = []
    for ip, name in ptr.items():
        if not name.startswith("swi"):
            continue
        base30, base31 = ip & ~3, ip & ~1
        net = ptr.get(base30, "")
        if net.startswith("net-"):          # /30 with a named network address
            peers = [ptr.get(base30 + k, "") for k in (1, 2) if base30 + k != ip]
        else:                               # /31
            peers = [ptr.get(base31 + (1 - (ip - base31)), "")]
        peer = next((p for p in peers if p), "")
        recs.append(dict(ip=str(ipaddress.ip_address(ip)), name=name, link=iface(name), peer_name=peer,
                         peer_link=iface(peer) if peer.startswith("swi") else None,
                         p2p_net=net if net.startswith("net-") else ""))
    by64: dict[int, list] = {}
    for ip, name in v6:
        by64.setdefault(int(ipaddress.ip_address(ip)) >> 64, []).append((ip, name))
    for ends in by64.values():
        for ip, name in ends:
            if not name.startswith("swi"):
                continue
            others = [n for i, n in ends if i != ip]
            peer = next((n for n in others if n.startswith("swi")), others[0] if others else "")
            recs.append(dict(ip=ip, name=name, link=iface(name), peer_name=peer,
                             peer_link=iface(peer) if peer.startswith("swi") else None, p2p_net=""))
    return pd.DataFrame(recs, columns=["ip", "name", "link", "peer_name", "peer_link", "p2p_net"])


def join(I: pd.DataFrame, L: pd.DataFrame) -> pd.DataFrame:
    """Add dns_name, dns_peer, dns_peer_name and dns_peer_reports (the far-end router is
    in the export) to the interfaces table; bundle members inherit the bundle's."""
    L = L[L.link.notna()].sort_values("peer_link", na_position="last").drop_duplicates("link")
    dn = L.set_index("link")
    I = I.copy()
    devs = set(I.dev)
    for c, src in (("dns_name", "name"), ("dns_peer", "peer_link"), ("dns_peer_name", "peer_name")):
        I[c] = I.link.map(dn[src])
        m = I[c].isna() & I.bundle.notna() & I.bundle.ne("")
        I.loc[m, c] = I.loc[m, "bundle"].map(dn[src])
    I["dns_peer_reports"] = I.dns_peer.str.split("/").str[0].isin(devs)
    return I


def live_routers(L: pd.DataFrame) -> set[str]:
    """Routers (short names) with at least one named interface: alive at sweep time."""
    L = L.dropna(subset=["link"])
    return {d for d in L.link.str.split("/").str[0] if "lab" not in d}


# --- sweeps -------------------------------------------------------------------

def _dig(args: list[str]) -> str:
    return subprocess.run(["dig", *args], capture_output=True, text=True).stdout


def sweep_v4(out: str | Path, net: str = V4_NET, workers: int = 8) -> int:
    """PTR of every address in ``net``; retries timeouts once more patiently.
    Writes ``<ip> <name>`` lines sorted by address; returns the number of names."""
    ips = [str(ip) for ip in ipaddress.ip_network(net)]

    def one(ip, timing=("+time=2", "+tries=1")):
        lines = _dig(["+short", *timing, "-x", ip]).splitlines()
        return lines[0] if lines else None        # ";; ..." = timed out
    with cf.ThreadPoolExecutor(workers) as ex:
        names = dict(zip(ips, ex.map(one, ips)))
    retry = [ip for ip, n in names.items() if n and n.startswith(";;")]
    with cf.ThreadPoolExecutor(max(1, workers // 2)) as ex:
        names.update(zip(retry, ex.map(lambda ip: one(ip, ("+time=4", "+tries=3")), retry)))
    names = {ip: n for ip, n in names.items() if n and not n.startswith(";;")}
    with open(out, "w") as f:
        for ip in sorted(names, key=lambda x: int(ipaddress.ip_address(x))):
            f.write(f"{ip} {names[ip]}\n")
    return len(names)


def walk_v6(out: str | Path, zone: str = V6_ZONE, depth: int = 4, workers: int = 8) -> int:
    """Walk ``zone`` (a /48) ``depth`` nibbles down (4 -> /64s), pruning on NXDOMAIN
    (RFC 8020: no names below a non-existent node), then read ::1 .. ::6 of each /64."""
    def exists(name):
        return "NXDOMAIN" not in _dig(["+time=3", "+tries=2", name, "PTR", "+noall", "+comments"])

    def ptr(name):
        r = _dig(["+short", "+time=3", "+tries=2", name, "PTR"]).split()
        return r[0] if r else None
    level = [zone]
    for d in range(depth):
        cand = [f"{h}.{p}" for p in level for h in "0123456789abcdef"]
        with cf.ThreadPoolExecutor(workers) as ex:
            ok = list(ex.map(exists, cand))
        level = [c for c, o in zip(cand, ok) if o]
        print(f"[dns] depth {d + 1}: {len(level)} existing", flush=True)
    names = [".".join(list(f"{k:016x}"[::-1])) + "." + p for p in level for k in range(1, 7)]
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(ptr, names))
    n = 0
    with open(out, "w") as f:
        for name, r in zip(names, res):
            if r:
                nib = name.replace(".ip6.arpa", "").split(".")[::-1]
                ip = ipaddress.ip_address(":".join("".join(nib[i:i + 4]) for i in range(0, 32, 4)))
                f.write(f"{ip} {r}\n")
                n += 1
    return n
