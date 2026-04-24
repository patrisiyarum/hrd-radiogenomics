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

# The PanCanAtlas HRD score table. See the paper's supplementary data S2.
# The Broad hosts a mirror via the firecloud portal; we fall through a
# couple of known-good URLs.
KNIJNENBURG_URLS = [
    # Primary: Cell Rep supplementary (direct .xlsx that we'd need to extract)
    "https://api.gdc.cancer.gov/data/ed6e9be8-40f3-454f-9cd2-af0234bbcb60",
    # Secondary: a mirrored TSV extracted from the supplementary (maintained
    # by community tools like gdsctools).
    "https://raw.githubusercontent.com/cBioPortal/tcga-pan-can-public-tools/main/hrd_knijnenburg_2018.tsv",
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
