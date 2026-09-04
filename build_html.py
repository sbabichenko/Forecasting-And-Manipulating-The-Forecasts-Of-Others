#!/usr/bin/env python3
"""
Build a self-contained interactive.html by inlining solver.js into the template.
Plotly.js is loaded from CDN (3.5MB is too large to inline).
"""

import os
import json

DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(DIR, "solver.js"), "r", encoding="utf-8") as f:
    solver_js = f.read()

with open(os.path.join(DIR, "interactive_template.html"), "r", encoding="utf-8") as f:
    template = f.read()

stationary_param_nn_path = os.path.join(DIR, "data", "stationary_param_pinn.json")
if not os.path.exists(stationary_param_nn_path):
    stationary_param_nn_path = os.path.join(DIR, "data", "stationary_param_nn.json")
if not os.path.exists(stationary_param_nn_path):
    stationary_param_nn_path = os.path.join(DIR, "data", "stationary_param_surrogate.json")
stationary_param_nn_json = "null"
if os.path.exists(stationary_param_nn_path):
    with open(stationary_param_nn_path, "r", encoding="utf-8") as f:
        stationary_param_nn_json = json.dumps(json.load(f), separators=(",", ":"))

kyle_back_equilibrium_path = os.environ.get("KYLE_BACK_EQUILIBRIUM_JSON")
if not kyle_back_equilibrium_path:
    kyle_back_equilibrium_path = os.path.join(DIR, "data", "kyle_back_irf_lag_pinn.json")
if not os.path.exists(kyle_back_equilibrium_path):
    kyle_back_equilibrium_path = os.path.join(DIR, "data", "kyle_back_fixed_policy_filter_N41.json")
if not os.path.exists(kyle_back_equilibrium_path):
    kyle_back_equilibrium_path = os.path.join(DIR, "data", "kyle_back_stationary_param_two_net_pinn_N41.json")
if not os.path.exists(kyle_back_equilibrium_path):
    kyle_back_equilibrium_path = os.path.join(DIR, "data", "kyle_back_stationary_param_two_net_pinn.json")
if not os.path.exists(kyle_back_equilibrium_path):
    kyle_back_equilibrium_path = os.path.join(DIR, "data", "kyle_back_stationary_param_pinn_template_fixed.json")
if not os.path.exists(kyle_back_equilibrium_path):
    kyle_back_equilibrium_path = os.path.join(DIR, "data", "kyle_back_equilibrium_pinn_pdf_birth.json")
if not os.path.exists(kyle_back_equilibrium_path):
    kyle_back_equilibrium_path = os.path.join(DIR, "data", "kyle_back_equilibrium_pinn_vtilde.json")
if not os.path.exists(kyle_back_equilibrium_path):
    kyle_back_equilibrium_path = os.path.join(DIR, "data", "kyle_back_equilibrium_pinn_atom_boundary.json")
if not os.path.exists(kyle_back_equilibrium_path):
    kyle_back_equilibrium_path = os.path.join(DIR, "data", "kyle_back_equilibrium_pinn_openquad_boundary.json")
if not os.path.exists(kyle_back_equilibrium_path):
    kyle_back_equilibrium_path = os.path.join(DIR, "data", "kyle_back_equilibrium_pinn_filter_continue.json")
if not os.path.exists(kyle_back_equilibrium_path):
    kyle_back_equilibrium_path = os.path.join(DIR, "data", "kyle_back_equilibrium_pinn.json")
kyle_back_equilibrium_json = "null"
if os.path.exists(kyle_back_equilibrium_path):
    with open(kyle_back_equilibrium_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    diag = payload.get("diagnostics")
    if isinstance(diag, dict):
        if "candidate_fixed_point_rms" in diag and "policy_target_rms" not in diag:
            diag["policy_target_rms"] = diag.pop("candidate_fixed_point_rms")
        # Keep the browser payload focused on the active checkpoint.  The saved
        # training JSON can carry nested best-checkpoint arrays and theorem
        # dumps, but the UI reads the top-level exported fields below.
        diag.pop("best", None)
        diag.pop("theorem_1_12_policy", None)
    compact = {
        key: payload[key]
        for key in (
            "version",
            "kind",
            "description",
            "assumptions",
            "mathematical_conventions",
            "architecture",
            "params",
            "derived_params",
            "grid",
            "fields",
            "policy_noise_state",
            "primitive_demand",
            "calD",
            "mode_decomposition",
            "noise_state_irfs",
            "value_irfs",
            "foc_terms",
            "filter_closure_residuals",
            "vtilde_projection_residuals",
            "zero_filter_residuals",
            "filter_tilde",
            "unresolved_value",
            "unresolved_total_demand",
            "diagnostics",
        )
        if key in payload
    }
    kyle_back_equilibrium_json = json.dumps(compact, separators=(",", ":"))

html = (template
        .replace("/* SOLVER_JS_INLINE */", solver_js)
        .replace("/* STATIONARY_PARAM_NN_INLINE */", stationary_param_nn_json)
        .replace("/* KYLE_BACK_EQUILIBRIUM_INLINE */", kyle_back_equilibrium_json))

out_path = os.environ.get("INTERACTIVE_OUT", os.path.join(DIR, "interactive.html"))
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize(out_path) / 1024
print(f"Built {out_path} ({size_kb:.0f} KB)")
