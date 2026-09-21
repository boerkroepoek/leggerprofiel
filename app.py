"""Streamlit-interface voor het ontwerpen van een leggerprofiel."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from leggerprofiel.model import (
    ProfileValidationError,
    build_profile,
    default_boundaries,
    default_layers,
    default_profile,
    to_excel_bytes,
)

st.set_page_config(page_title="Leggerprofiel", page_icon="📐", layout="wide")


def initialize_state() -> None:
    """Initialiseer bewerkbare invoer eenmalig per sessie."""
    defaults = {
        "profile": default_profile(),
        "layers": default_layers(),
        "boundaries": default_boundaries(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def profile_figure(result) -> go.Figure:
    """Bouw een interactieve dwarsdoorsnede."""
    points = result.combined
    surface = points[~points["Omschrijving"].astype(str).str.startswith(("Onderkant", "Grens"))]
    surface = surface.sort_values("X")
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=surface["X"], y=surface["Y"], mode="lines+markers+text",
        text=surface["Omschrijving"].str.extract(r"^(\d+)", expand=False),
        textposition="top center", name="Maaiveld", line={"color": "#334155", "width": 3},
        hovertemplate="%{customdata}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        customdata=surface["Omschrijving"],
    ))
    colors = {"Onderkant zand": "#d6a85f", "Onderkant veen": "#8b6b47", "Onderkant klei": "#7c8ca5"}
    for layer_name, group in points[points["Omschrijving"].astype(str).str.startswith("Onderkant")].groupby("Omschrijving", sort=False):
        group = group.sort_values("X")
        figure.add_trace(go.Scatter(
            x=group["X"], y=group["Y"], mode="lines+markers", name=layer_name,
            line={"color": colors.get(layer_name, "#64748b"), "width": 2},
        ))
    for _, boundary in result.boundaries.iterrows():
        figure.add_vline(x=boundary["X"], line_color="#ef4444", line_width=1.5)
        figure.add_annotation(x=boundary["X"], y=boundary["Y"], text=boundary["Omschrijving"],
                              textangle=-90, showarrow=False, yshift=45, font={"size": 10})
    figure.update_layout(
        height=620, margin={"l": 30, "r": 20, "t": 30, "b": 30},
        xaxis_title="Afstand X (m)", yaxis_title="Hoogte Y (m)",
        hovermode="closest", legend={"orientation": "h", "y": -0.15},
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


initialize_state()
st.title("Leggerprofiel")
st.caption("Interactieve omzetting van het Excel-rekenblad naar een reproduceerbare profielberekening.")

with st.sidebar:
    st.header("Acties")
    if st.button("Herstel standaardwaarden", use_container_width=True):
        st.session_state.profile = default_profile()
        st.session_state.layers = default_layers()
        st.session_state.boundaries = default_boundaries()
        st.rerun()
    st.info("Bewerk tabellen direct. Alle afgeleide waarden worden na iedere wijziging opnieuw berekend.")

tab_profile, tab_layers, tab_zones = st.tabs(["Profielpunten", "Grondopbouw", "Zonering"])
with tab_profile:
    st.session_state.profile = st.data_editor(
        st.session_state.profile, num_rows="dynamic", use_container_width=True, key="profile_editor",
        column_config={"X": st.column_config.NumberColumn(format="%.2f"), "Y": st.column_config.NumberColumn(format="%.2f")},
    )
with tab_layers:
    st.session_state.layers = st.data_editor(
        st.session_state.layers, num_rows="dynamic", use_container_width=True, key="layers_editor",
        column_config={"Laagscheiding": st.column_config.NumberColumn(format="%.2f"), "Helling": st.column_config.NumberColumn(min_value=0.0, format="%.2f")},
    )
with tab_zones:
    st.session_state.boundaries = st.data_editor(
        st.session_state.boundaries, num_rows="dynamic", use_container_width=True, key="boundaries_editor",
        column_config={"Zijde": st.column_config.SelectboxColumn(options=["Buitendijks", "Binnendijks"]), "X": st.column_config.NumberColumn(format="%.2f")},
    )

try:
    result = build_profile(st.session_state.profile, st.session_state.layers, st.session_state.boundaries)
except ProfileValidationError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Dwarsprofiel")
st.plotly_chart(profile_figure(result), use_container_width=True)

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Profielpunten", len(st.session_state.profile))
metric_2.metric("Profiellengte buitendijks", f"{result.outside_layers['Afstand'].sum():.2f} m")
metric_3.metric("Profiellengte binnendijks", f"{result.inside_layers['Afstand'].sum():.2f} m")

with st.expander("Berekende tabellen", expanded=False):
    left, right = st.columns(2)
    left.markdown("**Buitendijks**")
    left.dataframe(result.outside_layers, use_container_width=True, hide_index=True)
    right.markdown("**Binnendijks**")
    right.dataframe(result.inside_layers, use_container_width=True, hide_index=True)
    st.markdown("**Gecombineerde uitvoer**")
    st.dataframe(result.combined, use_container_width=True, hide_index=True)

excel = to_excel_bytes(result, st.session_state.profile, st.session_state.layers)
col_a, col_b = st.columns(2)
col_a.download_button("Download berekening als Excel", excel, "leggerprofiel_berekend.xlsx",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
col_b.download_button("Download gecombineerd profiel als CSV", result.combined.to_csv(index=False).encode("utf-8-sig"),
                      "leggerprofiel.csv", "text/csv", use_container_width=True)
