# Topology viewer

The SWITCH backbone map with traffic and router telemetry over time. Build the frames
from the data (S3 or a local mirror, through the `green_routing` package) and serve:

```bash
python build_frames.py      # -> data/frames/<YYYYMM>.json
python server.py            # http://localhost:8137
```

Full documentation: [docs/viewers.md](../docs/viewers.md#topology-viewer-web).
