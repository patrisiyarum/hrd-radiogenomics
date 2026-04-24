"""Fetch the Knijnenburg 2018 HRD-label supplementary table.

Cell Rep paper: https://doi.org/10.1016/j.celrep.2018.03.076
Supplementary: https://www.cell.com/cms/10.1016/j.celrep.2018.03.076/attachment/...

We mirror a single canonical copy to avoid the paper host going down
mid-pipeline. If the mirror URL rots, re-host from Cell Rep and update.
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

logger = logging.getLogger("download_knijnenburg")

# The PanCanAtlas HRD score table hosted by the GDC publications page.
# File: TCGA.HRD_withSampleID.txt — tab-separated, 10,648 rows, one per
# TCGA sample. Columns: sampleID, ai1 (NtAI count), lst1 (LST count),
# hrd-loh (HRD-LOH count), HRD (the summed score).
# Source:  https://gdc.cancer.gov/about-data/publications/panimmune
KNIJNENBURG_URLS = [
    "https://api.gdc.cancer.gov/data/66dd07d7-6366-4774-83c3-5ad1e22b177e",
]


def main() -> None:
    sm = globals().get("snakemake")
    logging.basicConfig(level=logging.INFO)
    if sm is None:
        raise SystemExit("run via snakemake")
    out = Path(sm.output.table)
    out.parent.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for url in KNIJNENBURG_URLS:
        try:
            logger.info("fetching %s", url)
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            out.write_bytes(r.content)
            logger.info("wrote %d bytes to %s", len(r.content), out)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed: %s", exc)
            last_err = exc
    raise RuntimeError(
        f"every Knijnenburg mirror failed; last error: {last_err}. "
        "Download the Cell Rep 2018 supplementary manually and place it at "
        f"{out}."
    )


if __name__ == "__main__":
    main()
