@AGENTS.md

## Notes for Claude Code sessions

- Activate the environment in every shell: `source ~/anaconda3/etc/profile.d/conda.sh && conda activate green`.
- Run anything longer than a few minutes detached (`setsid nohup … > log 2>&1 &`) and check
  its log or `run.log`; background tool tasks end with the session. Training and the
  recipes resume where they stopped.
- Never `pkill -f` a pattern that also appears in your own command line (it kills your
  shell); find the PID first.
- Quick checks after changes: `pytest -m "not e2e and not live"`; before committing:
  `pytest -m e2e`.
- Results claims: check them against the CSVs in the run directory before reporting.
