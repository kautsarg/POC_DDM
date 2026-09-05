# POC_DDM

Point-of-care data-driven multiplexing: turning raw Lacewing/Titan chip readouts
into trained classifiers and interpretability reports.


| Folder | Status | What it is |
|---|---|---|
| **`main_code/`** | **Finalised / production** | The cleaned, consolidated pipeline. Two dispatchers (`main_chip.py`, `main_lab.py`), a single `config.py`, and a fixed dataset scope. This is the code to read, run, and build on. |
| **`gk_code/main/`** | **Experimentation** | The exploratory tree the results were developed in — numbered stage scripts, ~90 analysis notebooks, ablations, and SLURM sweeps. Kept for provenance and for the thesis analyses; not the reference implementation. |
| `titan_v4/`, `titan_v6/` | Device layer | Raw chip acquisition/decoding for the v04 and v05/v06 firmware. `main_code/utils/titan_v6/` holds a synchronised copy. |
