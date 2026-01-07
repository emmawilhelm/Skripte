import math

# -----------------------------
# Konstanten
# -----------------------------
g = 9.81            # m/s^2
rho = 1000.0        # kg/m^3 (Wasser)
nu = 1.0e-6         # m^2/s (Wasser ~20°C; in deiner Sammlung angegeben)  # :contentReference[oaicite:2]{index=2}

# -----------------------------
# Geometrie (dein Fall)
# -----------------------------
D = 4.30            # Rohrdurchmesser [m]
d = 2.00            # Lochdurchmesser [m]
L_up = 3.0          # [m] vor Störstelle
L_down = 3.0        # [m] nach Störstelle
L = L_up + L_down   # Gesamtlänge im Ausschnitt

A_pipe = math.pi * (D**2) / 4.0
A_hole = math.pi * (d**2) / 4.0
beta = d / D

# -----------------------------
# Hilfsfunktionen
# -----------------------------
def reynolds(v, d_hy, nu_):
    return v * d_hy / nu_

def colebrook_white(Re, ks, D_, tol=1e-8, maxit=80):
    """
    Iterative Lösung der Colebrook-White Gleichung.
    Formeln für verschiedene Rauigkeitsbereiche stehen in deiner Sammlung.  :contentReference[oaicite:3]{index=3}
    Wir lösen allgemein die "Übergang"-Form:
      1/sqrt(lambda) = -2 log10( 2.51/(Re*sqrt(lambda)) + ks/(3.71*D) )
    """
    if Re < 1e-8:
        return 0.0

    # Startwert (Blasius als grober Start für turbulente/glatte Bereiche)
    lam = 0.02
    for _ in range(maxit):
        if lam <= 0:
            lam = 1e-6
        lhs = 1.0 / math.sqrt(lam)
        rhs = -2.0 * math.log10(2.51 / (Re * math.sqrt(lam)) + ks / (3.71 * D_))
        # einfache Fixpunktiteration
        lam_new = 1.0 / (rhs * rhs)
        if abs(lam_new - lam) < tol:
            return lam_new
        lam = lam_new
    return lam

def headloss_total(Q, ks, zeta_extra=0.0, zeta_orifice=0.0, orifice_ref="pipe"):
    """
    Berechnet Gesamtverlusthöhe hv = he + hr im 6m-Ausschnitt.
    - Rohrreibung: hr = (v^2/(2g))*lambda*(L/D)            :contentReference[oaicite:4]{index=4}
    - Einzelverluste: he = (v^2/(2g))*sum(zeta)            :contentReference[oaicite:5]{index=5}
    Lochscheibe:
      - v_pipe = Q/A_pipe
      - v_hole = Q/A_hole = v_pipe*(A_pipe/A_hole)
      ζ_orifice ist ein Eingabeparameter (du kannst ihn später aus Literatur/Versuch setzen).
    """
    v_pipe = Q / A_pipe
    v_hole = Q / A_hole

    Re_pipe = reynolds(v_pipe, D, nu)
    lam = colebrook_white(Re_pipe, ks, D)

    # Reibungsverlust im Ausschnitt
    hr = (v_pipe**2) / (2*g) * lam * (L / D)

    # Einzelverluste: Basisgeschwindigkeit wählen
    if orifice_ref == "hole":
        v_ref = v_hole
    else:
        v_ref = v_pipe

    he_orifice = (v_ref**2) / (2*g) * zeta_orifice
    he_other = (v_pipe**2) / (2*g) * zeta_extra

    hv = hr + he_orifice + he_other

    return {
        "hv": hv,
        "v_pipe": v_pipe,
        "v_hole": v_hole,
        "Re_pipe": Re_pipe,
        "lambda": lam,
        "hr": hr,
        "he_orifice": he_orifice,
        "he_other": he_other,
    }

def solve_Q_for_delta_h(delta_h, ks, zeta_extra=0.0, zeta_orifice=0.0, orifice_ref="pipe"):
    """
    Findet Q so, dass hv(Q) = delta_h.
    Nutzt einfache Bisektion (robust).
    """
    # grobe Schranken
    Q_lo = 0.0
    Q_hi = A_pipe * 10.0  # entspricht 10 m/s im Rohr als obere Schranke

    def f(Q):
        return headloss_total(Q, ks, zeta_extra, zeta_orifice, orifice_ref)["hv"] - delta_h

    # erweitere Q_hi bis f(Q_hi) > 0
    while f(Q_hi) < 0:
        Q_hi *= 1.5
        if Q_hi > A_pipe * 100:
            raise RuntimeError("Keine passende obere Schranke gefunden; prüfe delta_h/Parameter.")

    for _ in range(80):
        Q_mid = 0.5 * (Q_lo + Q_hi)
        if f(Q_mid) > 0:
            Q_hi = Q_mid
        else:
            Q_lo = Q_mid
    return 0.5 * (Q_lo + Q_hi)

# -----------------------------
# Beispiel-Run (bitte anpassen)
# -----------------------------
if __name__ == "__main__":
    # Rauheit ks [m] (aus deiner Tabelle typischerweise mm -> in m umrechnen)
    # Beispiel: handelsübliches Stahlrohr 0.05 mm -> 0.00005 m
    ks = 0.00005

    # Einzelverlust-Beiwert der Lochscheibe ζ (muss festgelegt werden)
    # Wenn du keinen Wert hast: starte z.B. mit 2..10 und kalibriere später.
    zeta_orifice = 0.6

    # weitere Einzelverluste im Ausschnitt (z.B. Ein-/Auslauf etc.)
    zeta_extra = 0.0

    # Vorgabe: Höhen-/Druckdifferenz als Verlusthöhe delta_h [m]
    delta_h = 1.0  # m

    Q = solve_Q_for_delta_h(delta_h, ks, zeta_extra, zeta_orifice, orifice_ref="pipe")
    res = headloss_total(Q, ks, zeta_extra, zeta_orifice, orifice_ref="pipe")

    print("\n=== Ergebnisse (Ausschnitt 3m + Orifice + 3m) ===")
    print(f"Rohr D = {D:.2f} m, Loch d = {d:.2f} m, beta = {beta:.4f}")
    print(f"Q        = {Q:.4f} m^3/s")
    print(f"v_pipe   = {res['v_pipe']:.4f} m/s")
    print(f"v_hole   = {res['v_hole']:.4f} m/s")
    print(f"Re_pipe  = {res['Re_pipe']:.3e}")
    print(f"lambda   = {res['lambda']:.5f}")
    print(f"hr       = {res['hr']:.4f} m")
    print(f"he_orif  = {res['he_orifice']:.4f} m")
    print(f"he_other = {res['he_other']:.4f} m")
    print(f"hv total = {res['hv']:.4f} m (soll ~ delta_h = {delta_h:.4f} m sein)")
