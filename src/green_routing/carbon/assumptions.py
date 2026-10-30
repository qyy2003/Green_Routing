"""Every number the carbon pipeline assumes rather than measures, in one place.

Change them here, not in the code that uses them; docs/carbon.md explains each choice.
"""
from __future__ import annotations

# --- Facility overhead (PUE) by host type of the site: (central, low, high).
# Not measured for any SWITCH site. Swiss surveys put colocation at 1.25-1.6 and internal
# data centres at ~1.8 (BFE 2014, asut 2023), small server rooms at 2.0-2.1 (BFE 2014);
# sources in the reference table overhead_contracts_embodied.csv, section A.
PUE = {
    "colocation": (1.3, 1.15, 1.6),
    "switch_own": (1.3, 1.15, 1.6),
    "university": (1.6, 1.3, 2.1),
    "research_institute": (1.6, 1.3, 2.1),
    "government": (1.6, 1.3, 2.1),
    "hospital": (1.8, 1.4, 2.1),
    "other": (1.6, 1.2, 2.1),
}
PUE_DEFAULT = (1.6, 1.2, 2.1)


def pue(host_type, which: int = 0) -> float:
    """PUE of a host type (0 central, 1 low, 2 high); unknown types get the default."""
    return PUE.get(host_type, PUE_DEFAULT)[which]


# --- Router power where the PSUs are not metered: (central, low, high) W, from
# datasheets scaled by the measured/datasheet ratio of metered peers, else literature.
# Routers with metered model peers use the peers' median (+-10 %) instead.
MODEL_W = {
    "ASR-920 (variant ?)": (73, 70, 86),
    "ASR-920-12CZ-A": (55, 50, 80),
    "ASR-920-4SZ-A": (55, 50, 75),
    "N540-28Z4C": (160, 130, 200),
    "N540X (small)": (40, 33, 60),
    "Catalyst 6500": (334, 334, 1000),
    "ME3400-class switch": (50, 30, 80),
    "NCS-55A1-24Q6H-SS/48Q6H": (310, 280, 350),
    "unknown": (150, 40, 400),
}
PSU_EFFICIENCY = 0.85      # UI-schema PSUs report DC output: wall power ~ W / 0.85
PIN_BAND_W = (20, 1.05)    # metered input power: low = W - 20, high = W * 1.05

# --- Pluggable optics, typical W per module by port speed (Gbit/s). Used to split a
# router's power between its customer-facing and its backbone ports.
TRANSCEIVER_W = {1: 1.0, 10: 1.5, 25: 1.5, 40: 3.5, 100: 4.0, 400: 12.0}
TRANSCEIVER_W_DEFAULT = 4.0       # unknown speed on a named port
TRANSCEIVER_W_NO_SPEED = 1.5      # no speed in the name
OPTICS_MAX_SHARE = 0.6            # optics never more than 60 % of a router's energy

# --- Grid zones. Every site draws from the Swiss grid except CERN, which is fed from
# the French side; Liechtenstein is in Swissgrid's control area.
SITE_ZONE = {"ce": "FR"}
DEFAULT_ZONE = "CH"


def zone_of(site: str) -> str:
    return SITE_ZONE.get(site, DEFAULT_ZONE)


# --- Border routers missing from the export: what is behind their trunks.
# From SWITCH's looking glass, PeeringDB and the GEANT Compendium (2026-10-30).
BORDER_ROUTERS = {
    "ix3": "external via ix3 (Arelion, SwissIX, Equinix IX, PNIs)",
    "ce1": "external via ce1 (GEANT Paris, CIXP, GTT, PNIs)",
    "ba3": "external via ba3 (DE-CIX, Cogent, Lumen, DFN, Init7)",
}

# --- Marginal energy per bit of one router, J/bit (Fantastic Joules, IMC'25: 3-22 pJ/bit),
# and routers on a typical SWITCHlan path. Only for the marginal g/GB comparison.
MARGINAL_J_PER_BIT = (3e-12, 22e-12)
ROUTERS_PER_PATH = 3

GB_PER_MBPS_HOUR = 3600 / 8 / 1000     # a mean of 1 Mbit/s over an hour = 0.45 GB
