# Cold Compressibility Cpk / Ppk Capability App

A deployable Streamlit application for brake-pad cold-compressibility capability analysis in µm. The package targets Streamlit 1.61.0 (released August 4, 2026).

## What it does

- Upload `.xlsx`, `.xls`, `.csv`, or `.txt` measurement files.
- Select the Excel sheet and measurement column.
- Paste a copied Excel column (including German decimal comma values).
- Manually enter/edit any number of values in a dynamic table.
- No fixed pad count: 50, 100, 500, or more values work without changing formulas.
- Editable nominal/tolerance or direct LSL/USL specifications.
- Calculates mean, STDEV.S, Cp, Cpk, Pp, Ppk, tail indices, outside-spec count, and predicted nonconforming ppm.
- Default **Direct STDEV.S** mode matches the supplied Excel tool: Cpk and Ppk use the same sample standard deviation.
- Optional **I-MR** mode estimates within-process sigma as MR-bar / 1.128, so Cpk and Ppk can differ.
- Interactive measurement-sequence scatter plot with LSL/USL/target/mean lines.
- Histogram + fitted normal distribution with specification window and ±3σ width.
- Capability “what-if” plot to demonstrate the effect of moving the mean or reducing process variation.
- Download an Excel capability report and cleaned CSV data.

## Run locally

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The browser will normally open automatically. If not, use the local URL printed by Streamlit.

## Deploy online with Streamlit Community Cloud

1. Create a GitHub repository and put all files from this folder in the repository.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Choose **Create app / Deploy an app**.
4. Select the repository, branch, and `app.py` as the entrypoint.
5. Deploy. Streamlit installs the packages from `requirements.txt` automatically.
6. Choose whether the app should be public or private according to your data/security needs.

### Important for company/supplier data

If measurements are confidential, do **not** deploy the tool publicly. Use a private repository/private Streamlit app, or deploy the same Streamlit app internally with Docker on an approved corporate server.

## Statistical note

### Direct STDEV.S mode
This follows the current supplied workbook logic:

- `s = STDEV.S(all measurements)`
- `CPL = (Mean - LSL) / (3s)`
- `CPU = (USL - Mean) / (3s)`
- `Cpk = min(CPL, CPU)`
- `PPL = (Mean - LSL) / (3s)`
- `PPU = (USL - Mean) / (3s)`
- `Ppk = min(PPL, PPU)`

Because both indices use the same `s`, Cpk and Ppk are identical in this mode.

### I-MR mode
For sequential individual measurements without subgroups:

- overall sigma = `STDEV.S(all measurements)` → Ppk
- moving range = `abs(X[i] - X[i-1])`
- within sigma = `average moving range / 1.128` → Cpk

Use the method that matches your agreed process-capability procedure and customer/supplier standard.
