Run Milky-Way processing

Assumes local DATA layout at ../../DATA relative to this script.

Run with system Python:

```bash
python3 milky-way/Milky-Way.py
```

Or with the project venv:

```bash
./env-extinction/bin/python milky-way/Milky-Way.py
```

Flags:

- `--force` : recompute and overwrite cached .npy files
- `--resume`: reuse existing .npy caches when available (default behavior)

Examples:

```bash
# recompute everything
python3 milky-way/Milky-Way.py --force

# reuse caches
python3 milky-way/Milky-Way.py --resume
```