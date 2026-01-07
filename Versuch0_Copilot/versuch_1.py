
from epb_slit_flow_sim import SlitOpening, ScrewParams, SimulationInput, solve_pc

from calibrate_epb_slit import calibrate_hb_from_flow_table

calib = calibrate_hb_from_flow_table(
    "2020-07-21 Versuchsprotokoll Flow table.xlsx",
    lambda_f=[0.35, 0.42, 0.28]  # optional: Liste aus ATUR-λf
)

tau0 = calib["tau0"]          # Pa
K    = calib["K"]             # Pa·s^n
n    = calib["n"]             # -
#

# Beispielparameter (bitte projekt-/labordatenbasiert anpassen)
sim = SimulationInput(
    p_face = 150_000.0,   # Ortsbrustdruck [Pa], z.B. 1.5 bar
    rho    = 1_600.0,     # Dichte [kg/m3]
    tau0   = 1_500.0,     # Fließgrenze [Pa]
    K      = 500.0,       # Konsistenz [Pa·s^n]
    n      = 0.45,        # Verhaltensindex [-]
    openings = [
        SlitOpening(b=0.30, h=0.02,  L=0.05, K_m_in=0.7, K_m_out=1.3, name="Schlitz-1"),
        SlitOpening(b=0.20, h=0.015, L=0.04, K_m_in=0.5, K_m_out=1.0, name="Schlitz-2"),
    ],
    omega = 12.0,         # Schneckendrehzahl [rad/s]
    screw = ScrewParams(k1=0.8/1000.0, k2=0.02/1_000_000.0) # grobe Platzhalter, kalibrieren!
)

result = solve_pc(sim)  # findet p_c*, summiert Q_open und koppelt mit Q_screw
print(result)
