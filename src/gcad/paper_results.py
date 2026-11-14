"""Numbers reported in the paper (Tables 1-3), kept for side-by-side comparison only.

Source: Liu, Gao & Jiao, "GCAD: Anomaly Detection in Multivariate Time Series from the Perspective
of Granger Causality", AAAI-25. These are *published* values, not outputs of this code base.
"""

from __future__ import annotations

DATASETS = ("swat", "smd", "msl", "smap", "psm")

# method -> dataset -> (AUROC, AUPRC)
TABLE2: dict[str, dict[str, tuple[float, float]]] = {
    "DAGMM": {
        "swat": (0.7882, 0.4955),
        "smd": (0.7516, 0.3988),
        "msl": (0.6209, 0.0163),
        "smap": (0.5989, 0.0380),
        "psm": (0.6556, 0.3860),
    },
    "USAD": {
        "swat": (0.8318, 0.7173),
        "smd": (0.9274, 0.5316),
        "msl": (0.5601, 0.0108),
        "smap": (0.5314, 0.0279),
        "psm": (0.6584, 0.4924),
    },
    "GDN": {
        "swat": (0.8493, 0.6076),
        "smd": (0.9006, 0.5655),
        "msl": (0.3846, 0.0055),
        "smap": (0.5115, 0.0338),
        "psm": (0.7284, 0.4964),
    },
    "AT": {
        "swat": (0.5117, 0.1851),
        "smd": (0.1765, 0.0576),
        "msl": (0.5391, 0.0141),
        "smap": (0.3467, 0.0188),
        "psm": (0.5058, 0.2902),
    },
    "GANF": {
        "swat": (0.8112, 0.3557),
        "smd": (0.6384, 0.0017),
        "msl": (0.6402, 0.0291),
        "smap": (0.6926, 0.5259),
        "psm": (0.6335, 0.4090),
    },
    "MEMTO": {
        "swat": (0.7799, 0.6067),
        "smd": (0.5042, 0.1187),
        "msl": (0.7271, 0.0654),
        "smap": (0.4791, 0.0176),
        "psm": (0.5026, 0.2913),
    },
    "GCAD": {
        "swat": (0.8690, 0.7758),
        "smd": (0.9533, 0.7502),
        "msl": (0.7658, 0.3679),
        "smap": (0.7273, 0.4555),
        "psm": (0.7618, 0.6136),
    },
}

# Table 3 ablation: variant -> dataset -> (AUROC, AUPRC)
TABLE3: dict[str, dict[str, tuple[float, float]]] = {
    "gcad": {"swat": (0.8690, 0.7758), "smd": (0.9533, 0.7502)},
    "no_spars": {"swat": (0.8559, 0.7676), "smd": (0.9531, 0.6915)},
    "no_gc": {"swat": (0.8478, 0.7513), "smd": (0.9405, 0.6424)},
    "no_tc": {"swat": (0.8400, 0.7558), "smd": (0.9498, 0.6748)},
}

# Table 1: dataset statistics (channels, train length, test length, anomaly ratio %)
TABLE1: dict[str, tuple[int, int, int, float]] = {
    "swat": (51, 47_520, 44_991, 12.20),
    "smd": (38, 28_479, 28_479, 9.46),
    "msl": (55, 3_682, 2_856, 0.74),
    "smap": (25, 2_876, 8_579, 2.14),
    "psm": (25, 132_481, 87_841, 27.76),
}
