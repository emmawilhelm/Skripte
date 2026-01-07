# epb_slit_flow_sim.py
# -*- coding: utf-8 -*-
"""
Simulation des Materialflusses eines Herschel–Bulkley-Fluids (konditionierter Erdbrei)
durch Schlitz-Öffnungen eines EPB-Schneidrads und Kopplung mit einer einfachen
Schnecken-Fördercharakteristik.

Autor: Emma Wilhelm (BabEng) – erstellt mit Unterstützung von M365 Copilot
Datum: 2025-12-14
Lizenz: MIT

Beschreibung
-----------
Dieses Modul bietet Funktionen, um den Volumenstrom Q durch einzelne Schlitzöffnungen
unter einer gegebenen Druckdifferenz zu berechnen (laminare, isotherme, vollentwickelte
Strömung, kurzkanalige Annäherung) und iterativ die Kammerdrucklage p_c zu finden,
so dass die Massenerhaltung zwischen Zufluss (alle Öffnungen) und Abfluss (Schnecke)
gewahrt bleibt.

Rheologie: Herschel–Bulkley
    tau = tau0 + K * (gamma_dot)**n   für tau >= tau0
    gamma_dot = 0                     für tau < tau0 (Plug-Zone)

Geometrie des Schlitzes
    Breite b [m], Höhe h [m], Länge L [m]
    Hydraulischer Durchmesser D_h = 2*b*h / (b + h)

Druckverluste an Ein-/Auslass (Minor Losses)
    Delta p_minor = 0.5 * rho * (v**2) * (K_m_in + K_m_out)
    v = Q / A, A = b*h

Schnecke (vereinfachte Kennlinie, kalibrierbar)
    Q_screw(omega, p_c) = max(0, k1*omega - k2*(p_c - p_face))
    (Parameter k1, k2 projekt-/maschinen-spezifisch aus Messdaten zu kalibrieren)

Hinweis
-------
Die Rheoparameter tau0, K, n sind aus BabEng-Labordaten (ATUR + Flow-Table)
abzuleiten. Dieses Modul stellt bewusst eine numerisch robuste, nachvollziehbare
Integration bereit und ist so gestaltet, dass eine weitere KI unkompliziert darauf
aufbauen oder es erweitern kann (z. B. CFD-Kopplung, DEM, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import json

# -----------------------------
# Datenklassen für Eingaben
# -----------------------------

@dataclass
class SlitOpening:
    """Schlitz-Öffnung.

    Attributes:
        b: Breite [m]
        h: Höhe [m]
        L: Länge [m]
        K_m_in: Minor-Loss-Koeffizient Einlauf [-]
        K_m_out: Minor-Loss-Koeffizient Auslauf [-]
        name: optionale Bezeichnung
    """
    b: float
    h: float
    L: float
    K_m_in: float = 0.0
    K_m_out: float = 0.0
    name: str = ""

    @property
    def area(self) -> float:
        return self.b * self.h

    @property
    def Dh(self) -> float:
        # Hydraulischer Durchmesser für Rechteckkanal
        return 2.0 * self.b * self.h / (self.b + self.h)

@dataclass
class ScrewParams:
    """Parameter der vereinfachten Schnecken-Kennlinie."""
    k1: float  # [m^3/s per rad/s], Durchsatzkoeffizient
    k2: float  # [m^3/(s·Pa)], Druckempfindlichkeit

@dataclass
class SimulationInput:
    p_face: float          # [Pa], Druck an der Ortsbrust
    rho: float             # [kg/m^3], Dichte des Erdbreis
    tau0: float            # [Pa], Fließgrenze
    K: float               # [Pa·s^n], Konsistenzindex
    n: float               # [-], Verhaltensindex
    openings: List[SlitOpening]
    omega: float           # [rad/s], Schneckendrehzahl
    screw: ScrewParams

# -----------------------------
# HB-Integration: Planarspalt
# -----------------------------

def hb_slit_Q(dp: float, L: float, b: float, h: float, tau0: float, Kc: float, n: float,
               grid_points: int = 600) -> Tuple[float, Dict[str, float]]:
    """Berechne Volumenstrom Q eines HB-Fluids durch einen Planarspalt (Schlitz).

    Annahmen: laminar, isotherm, vollentwickelt, stationär. Symmetrischer Kanal
    mit Halbhöhe H = h/2; Wand bei y = ±H, Zentrum bei y = 0.

    Wir integrieren das Profil v(y) ausgehend von der Wand (v=0) zur Mitte.

    Args:
        dp: wirksame Druckdifferenz über den Schlitz [Pa]
        L: Länge des Schlitzes [m]
        b: Breite [m]
        h: Höhe [m]
        tau0: Fließgrenze [Pa]
        Kc: Konsistenzindex [Pa·s^n]
        n: Verhaltensindex [-]
        grid_points: Diskretisierung der Halbhöhe

    Returns:
        (Q, diagnostics)
        Q: Volumenstrom [m^3/s]
        diagnostics: Dict mit Plug-Dicke, max. Scherrate etc.
    """
    if dp <= 0:
        return 0.0, {"plug_thickness": h/2.0, "gamma_max": 0.0}

    H = h / 2.0
    # Diskretisierung von Wand (y=H) zur Mitte (y=0)
    ys = [H * i / (grid_points - 1) for i in range(grid_points)]  # 0..H
    # Integration von H -> 0 (Richtungswechsel)
    ys_rev = list(reversed(ys))

    # Schubspannung-Verteilung: tau(y) = dp/(2L) * y
    coeff = dp / (2.0 * L)

    # Numerische Integration von dv/dy = -gamma_dot(y)
    # Randbedingung: v(H) = 0. Wir integrieren von H -> 0.
    v_at_y = [0.0 for _ in ys_rev]

    def gamma_dot(tau: float) -> float:
        if tau <= tau0:
            return 0.0
        return ((tau - tau0) / Kc) ** (1.0 / n)

    for idx in range(len(ys_rev) - 2, -1, -1):
        y_next = ys_rev[idx]
        y_curr = ys_rev[idx + 1]
        dy = y_curr - y_next
        tau_curr = coeff * y_curr
        tau_next = coeff * y_next
        g_curr = gamma_dot(tau_curr)
        g_next = gamma_dot(tau_next)
        dv = -0.5 * (g_curr + g_next) * dy
        v_at_y[idx] = v_at_y[idx + 1] + dv

    # Q = 2*b*∫_0^H v(y) dy
    integral = 0.0
    for idx in range(len(ys_rev) - 1):
        y0, y1 = ys_rev[idx], ys_rev[idx + 1]
        v0, v1 = v_at_y[idx], v_at_y[idx + 1]
        integral += 0.5 * (v0 + v1) * (y1 - y0)
    Q = 2.0 * b * integral

    y0_plug = min(H, (2.0 * L * tau0) / dp)
    gamma_max = gamma_dot(coeff * H)
    diagnostics = {
        "plug_thickness": y0_plug,
        "gamma_max": gamma_max,
        "v_center": v_at_y[-1],
        "v_wall": 0.0,
    }
    return Q, diagnostics

# --------------------------------------
# Öffnung mit Minor-Loss-Iterationen
# --------------------------------------

def opening_Q_slit(opening: SlitOpening, p_c: float, p_face: float, rho: float,
                   tau0: float, Kc: float, n: float,
                   tol_Q: float = 1e-8, max_iters: int = 50) -> Tuple[float, Dict[str, Any]]:
    A = opening.area
    dp_nom = p_c - p_face
    if dp_nom <= 0:
        return 0.0, {"iters": 0, "dp_eff": 0.0}

    Q, diag = hb_slit_Q(dp_nom, opening.L, opening.b, opening.h, tau0, Kc, n)
    for k in range(max_iters):
        v = Q / A if A > 0 else 0.0
        dp_minor = 0.5 * rho * v * v * (opening.K_m_in + opening.K_m_out)
        dp_eff = max(1.0, dp_nom - dp_minor)
        Q_new, diag = hb_slit_Q(dp_eff, opening.L, opening.b, opening.h, tau0, Kc, n)
        if abs(Q_new - Q) < tol_Q:
            return Q_new, {"iters": k + 1, "dp_eff": dp_eff, **diag}
        Q = Q_new
    return Q, {"iters": max_iters, "dp_eff": dp_nom, **diag}

# --------------------------------------
# Schneckenmodell & Massenbilanz
# --------------------------------------

def Q_screw(omega: float, p_c: float, p_face: float, params: ScrewParams) -> float:
    return max(0.0, params.k1 * omega - params.k2 * (p_c - p_face))


def sum_openings_Q(p_c: float, sim: SimulationInput) -> Tuple[float, List[Dict[str, Any]]]:
    total = 0.0
    diags = []
    for op in sim.openings:
        Qi, d = opening_Q_slit(op, p_c, sim.p_face, sim.rho, sim.tau0, sim.K, sim.n)
        total += Qi
        diags.append({"name": op.name, "Q": Qi, **d})
    return total, diags


def solve_pc(sim: SimulationInput,
             pc_low: Optional[float] = None,
             pc_high: Optional[float] = None,
             tol_pc: float = 50.0,
             max_iters: int = 60) -> Dict[str, Any]:
    p_face = sim.p_face

    def F(pc: float) -> float:
        Q_open, _ = sum_openings_Q(pc, sim)
        Q_sc = Q_screw(sim.omega, pc, p_face, sim.screw)
        return Q_open - Q_sc

    if pc_low is None:
        pc_low = p_face + 1.0e4
    if pc_high is None:
        pc_high = p_face + 5.0e5
        f_low = F(pc_low)
        f_high = F(pc_high)
        expand_count = 0
        while f_low * f_high > 0 and expand_count < 12:
            pc_high += 5.0e5
            f_high = F(pc_high)
            expand_count += 1
        if f_low * f_high > 0:
            raise RuntimeError("Kein gültiges Druck-Bracket gefunden – Kennlinie prüfen/kalibrieren.")

    f_low = F(pc_low)
    f_high = F(pc_high)
    if f_low * f_high > 0:
        raise RuntimeError("Ungültiges Bracket: F(pc_low) und F(pc_high) haben gleiches Vorzeichen.")

    iters = 0
    while (pc_high - pc_low) > tol_pc and iters < max_iters:
        iters += 1
        pc_mid = 0.5 * (pc_low + pc_high)
        f_mid = F(pc_mid)
        if f_low * f_mid <= 0:
            pc_high = pc_mid
            f_high = f_mid
        else:
            pc_low = pc_mid
            f_low = f_mid

    pc_star = 0.5 * (pc_low + pc_high)
    Q_open_star, diags = sum_openings_Q(pc_star, sim)
    Q_sc_star = Q_screw(sim.omega, pc_star, p_face, sim.screw)

    return {
        "p_c": pc_star,
        "Q_open_total": Q_open_star,
        "Q_screw": Q_sc_star,
        "residual": Q_open_star - Q_sc_star,
        "iterations": iters,
        "diags": diags,
    }

# -----------------------------
# JSON I/O
# -----------------------------

def from_json(data: Dict[str, Any]) -> SimulationInput:
    openings = [SlitOpening(**op) for op in data["openings"]]
    screw = ScrewParams(**data["screw"])
    return SimulationInput(
        p_face=data["p_face"],
        rho=data["rho"],
        tau0=data["tau0"],
        K=data["K"],
        n=data["n"],
        openings=openings,
        omega=data["omega"],
        screw=screw,
    )


def to_json(result: Dict[str, Any]) -> str:
    return json.dumps(result, indent=2)

