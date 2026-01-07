# calibrate_epb_slit.py
# -*- coding: utf-8 -*-
"""
Kalibrierung der Herschel–Bulkley-Parameter (tau0, K, n) aus BabEng-Flow-Table-Daten.

Hinweis: Dies ist eine pragmatische, empirische Näherung für EPB-Anwendungen.
Sie nutzt das Ausbreitmaß nach 40 Schlägen (D40, zwei Richtungen) als
Flowability-Indikator und leitet daraus tau0 und K ab. Optional kann
ATUR (λ_f) die Scherverdünnung n modulieren.

Eingabe: Excel (.xlsx) gemäß den BabEng-Templates "Versuchsprotokoll Flow table"
Spalten (typisch):
 - Durchmesser m40 Richtung 1 [cm]
 - Durchmesser m40 Richtung 2 [cm]

Optionale ATUR-Eingabe:
 - λ_f (finaler Verklebungsparameter) als Skalar (0..1) oder Liste; falls Liste,
   wird der Median verwendet.

Ausgabe: dict mit tau0 [Pa], K [Pa·s^n], n [-], sowie Statistik zu D40.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List

# Erwartete Spaltennamen (Varianten)
D40_COLS = [
    'Durchmesser m40 Richtung 1', 'Durchmesser m40 Richtung 2',
    'Durchmesser m40 R1', 'Durchmesser m40 R2',
]

def _load_flow_table_xlsx(xlsx_path: str) -> pd.DataFrame:
    # Lesen mit openpyxl
    df_all = pd.read_excel(xlsx_path, engine='openpyxl', sheet_name=None)
    # Heuristik: Sheet mit "Printout" oder dem meisten numerischen Inhalt
    candidate = None
    max_numeric = -1
    for name, df in df_all.items():
        numeric_count = df.select_dtypes(include=[np.number]).size
        if numeric_count > max_numeric:
            max_numeric = numeric_count
            candidate = df
    if candidate is None:
        raise ValueError('Keine verwertbaren Datenblätter gefunden.')
    # Spaltennamen bereinigen
    candidate.columns = [str(c).strip() for c in candidate.columns]
    return candidate


def _extract_d40_values(df: pd.DataFrame) -> List[float]:
    # Finde vorhandene D40-Spalten
    present = [c for c in D40_COLS if c in df.columns]
    if len(present) < 2:
        # Versuche generische Suche
        present = [c for c in df.columns if ('m40' in c.lower()) and ('durchmesser' in c.lower())]
    if len(present) < 2:
        raise ValueError('D40-Spalten nicht gefunden. Bitte Vorlage prüfen.')
    # Werte in cm -> in Meter umrechnen später
    d1 = pd.to_numeric(df[present[0]], errors='coerce')
    d2 = pd.to_numeric(df[present[1]], errors='coerce')
    d40 = np.nanmean(np.vstack([d1.values, d2.values]), axis=0)
    d40 = d40[~np.isnan(d40)]
    # Filter unrealistischer Einträge
    d40 = d40[(d40 > 5) & (d40 < 40)]
    return d40.tolist()


def calibrate_hb_from_flow_table(xlsx_path: str, lambda_f: Optional[List[float]] = None) -> Dict[str, Any]:
    df = _load_flow_table_xlsx(xlsx_path)
    d40_cm = _extract_d40_values(df)
    if len(d40_cm) == 0:
        raise ValueError('Keine gültigen D40-Werte gefunden.')
    d40_med = float(np.median(d40_cm))

    # Empirische Abbildung: höhere D40 => niedrigere tau0 & niedrigere K
    # Parameterwahl an EPB-Praxis angelehnt (Flow40 Ziel 15–20 cm):
    # tau0 ≈ 2800 - 65 * D40_cm  [Pa]
    # K    ≈ 800  - 25 * D40_cm  [Pa·s^n]
    tau0 = max(300.0, 2800.0 - 65.0 * d40_med)
    K    = max(80.0, 800.0  - 25.0 * d40_med)

    # Basis-Scherverdünnung
    n_base = 0.45
    n = n_base
    if lambda_f is not None:
        # Verwende Median von λ_f (0..1), höhere λ_f => stärkere Scherverdünnung (kleineres n)
        lam = float(np.median([v for v in lambda_f if (v is not None) and (0 <= v <= 1)])) if len(lambda_f) else None
        if lam is not None:
            n = float(np.clip(n_base - 0.15 * (lam - 0.35), 0.3, 0.8))

    return {
        'tau0': tau0,
        'K': K,
        'n': n,
        'd40_cm_median': d40_med,
        'd40_cm_min': float(np.min(d40_cm)),
        'd40_cm_max': float(np.max(d40_cm)),
        'count': len(d40_cm),
    }

if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--flow_xlsx', required=True)
    ap.add_argument('--lambda_f', type=float, nargs='*')
    args = ap.parse_args()
    res = calibrate_hb_from_flow_table(args.flow_xlsx, args.lambda_f)
    print(json.dumps(res, indent=2))
