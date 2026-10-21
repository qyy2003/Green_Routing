"""Clean cost benchmark (run with the GPU otherwise idle): week vs month configuration
per model at batch 85. Writes $GR_ARTIFACTS_DIR/exp/results/cost.json: parameters (M),
median train-step ms (fwd+bwd+Adam), peak GiB, epochs actually run (from each run's
history summary)."""
import json, time
from pathlib import Path
import numpy as np, torch
from green_routing.paths import ARTIFACTS_DIR as A
from green_routing.prediction.run import build_models, apply_overrides
from green_routing.prediction.data import load_run_panel
from green_routing.prediction.calendar_features import build_calendar
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
p = load_run_panel(A / "pred_year" / "merged")
cal = build_calendar(p.timestamps); hs = (3, 12, 96, 288, 2016)
def epochs(run_dir, m):
    f = run_dir / "history" / f"{m}.summary.json"
    return json.loads(f.read_text())["fit"]["epochs_run"] if f.exists() else None
out = {}
for tag, lb, cfg, rdir in [("week", 2016, {}, None),
        ("month", 8640, dict(batch_size=85, tcn_max_levels=12, patchtst_patch=64, patchtst_stride=32), None)]:
    ms = build_models([4, 5], hs, False, p, lb, 100, under_weight=2.0, calendar=True)
    apply_overrides(ms, cfg)
    for m in ms:
        m.batch_size = 85
        m._n_links = p.L; m._cal = cal if m.use_calendar else None
        net = m._build_net(lb, m.max_h).cuda().train()
        opt = torch.optim.Adam(net.parameters(), 1e-4)
        x = torch.randn(85, lb, device="cuda"); y = torch.randn(85, m.max_h, device="cuda")
        lid = torch.randint(0, p.L, (85,), device="cuda"); aid = torch.randint(80000, 100000, (85,), device="cuda")
        f = (lambda: net(x)) if m.name == "dlinear" else ((lambda: net(x, lid, aid)) if m.use_calendar else (lambda: net(x, lid)))
        torch.cuda.reset_peak_memory_stats(); ts = []
        for i in range(13):
            torch.cuda.synchronize(); t = time.time()
            opt.zero_grad(); loss = torch.nn.functional.smooth_l1_loss(f(), y); loss.backward(); opt.step()
            torch.cuda.synchronize(); ts.append(time.time() - t)
        key = f"{m.name}@{tag}"
        if tag == "week":
            ep = epochs(A / "pred_year" / {"gru": "part_gru", "tcn": "part_tcn", "dlinear": "part_dlinear_patchtst", "patchtst": "part_dlinear_patchtst"}[m.name], m.name)
        else:
            ep = epochs(A / "exp" / f"month_{m.name}", m.name)
        out[key] = {"params_m": sum(q.numel() for q in net.parameters()) / 1e6,
                    "ms_step": 1000 * float(np.median(ts[3:])),
                    "gib": torch.cuda.max_memory_allocated() / 2**30, "epochs": ep}
        print(key, {k: round(v, 2) if isinstance(v, float) else v for k, v in out[key].items()}, flush=True)
        del net, opt, x, y; torch.cuda.empty_cache()
out["tcn@wk20"] = {**out["tcn@week"], "epochs": epochs(A / "exp" / "wk20_tcn", "tcn")}
(A / "exp" / "results" / "cost.json").write_text(json.dumps(out, indent=1))
