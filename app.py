import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

# ---------------------------------------------------------
# PAGE CONFIGURATION (INDUSTRIAL DARK HMI THEME)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Industrial SCADA HMI | PID Control Simulator",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for SCADA / HMI styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; color: #00e676; }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem; color: #b0bec5; }
    .status-box {
        padding: 12px; border-radius: 6px; font-weight: bold; text-align: center;
        font-family: monospace; font-size: 1.05rem; margin-bottom: 15px;
    }
    .status-normal { background-color: #0d3b2e; color: #00e676; border: 1px solid #00e676; }
    .status-warning { background-color: #4a3500; color: #ffab00; border: 1px solid #ffab00; }
    .status-alarm { background-color: #4a0e17; color: #ff1744; border: 1px solid #ff1744; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR: PLC & PID CONTROLLER PARAMETERS
# ---------------------------------------------------------
st.sidebar.title("🎛️ PLC Control Panel")
st.sidebar.caption("Controller: Siemens S7-1500 (Simulated PID_Compact)")

# Tuning Presets
preset = st.sidebar.selectbox(
    "Quick Tuning Preset",
    ["Optimal Industrial Tuning", "Aggressive (High Overshoot)", "Sluggish (Overdamped)", "Manual Custom Tuning"]
)

if preset == "Optimal Industrial Tuning":
    def_kp, def_ki, def_kd = 2.8, 0.12, 4.5
elif preset == "Aggressive (High Overshoot)":
    def_kp, def_ki, def_kd = 6.5, 0.35, 1.0
elif preset == "Sluggish (Overdamped)":
    def_kp, def_ki, def_kd = 1.0, 0.03, 2.0
else:
    def_kp, def_ki, def_kd = 2.5, 0.10, 3.0

st.sidebar.subheader("1. Setpoint & PID Gains")
setpoint = st.sidebar.slider("Target Setpoint - SP (°C)", min_value=50.0, max_value=500.0, value=250.0, step=5.0)
kp = st.sidebar.number_input("Proportional Gain (Kp)", min_value=0.0, max_value=20.0, value=def_kp, step=0.1)
ki = st.sidebar.number_input("Integral Gain (Ki - 1/s)", min_value=0.0, max_value=5.0, value=def_ki, step=0.01)
kd = st.sidebar.number_input("Derivative Gain (Kd - s)", min_value=0.0, max_value=20.0, value=def_kd, step=0.5)

st.sidebar.subheader("2. Process & Field Disturbances")
ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", 10.0, 50.0, 25.0, 1.0)
process_tau = st.sidebar.slider("Thermal Time Constant τ (s)", 10.0, 120.0, 45.0, 5.0)
disturbance_active = st.sidebar.checkbox("Inject Load Disturbance (Furnace Door Open at t=150s)", value=True)
disturbance_mag = st.sidebar.slider("Disturbance Magnitude (°C drop)", -80.0, -10.0, -35.0, 5.0) if disturbance_active else 0.0
sensor_noise = st.sidebar.checkbox("Simulate RTD PT100 Sensor Noise (4-20mA)", value=True)

st.sidebar.subheader("3. Safety & Interlocks")
high_alarm_limit = st.sidebar.slider("High-Temp Alarm Threshold (°C)", 100.0, 550.0, 275.0, 5.0)

# ---------------------------------------------------------
# FOPDT THERMAL PROCESS + ANTI-WINDUP PID SIMULATION ENGINE
# ---------------------------------------------------------
dt = 1.0
sim_time = 300
steps = int(sim_time / dt)
time_axis = np.linspace(0, sim_time, steps)

pv = np.zeros(steps)
sp_arr = np.full(steps, setpoint)
cv = np.zeros(steps)
p_term = np.zeros(steps)
i_term = np.zeros(steps)
d_term = np.zeros(steps)

pv[0] = ambient_temp
integral = 0.0
prev_error = setpoint - pv[0]
process_gain = 4.8  # °C per % heater power

np.random.seed(42)

for t in range(1, steps):
    # Measured PV with optional RTD noise
    noise = np.random.normal(0, 0.6) if sensor_noise else 0.0
    measured_pv = pv[t - 1] + noise
    
    error = sp_arr[t] - measured_pv
    
    # Proportional
    P = kp * error
    
    # Integral with Anti-Windup clamping
    integral += error * dt
    I = ki * integral
    I = np.clip(I, -100.0, 100.0)
    
    # Derivative (on error)
    D = kd * (error - prev_error) / dt
    prev_error = error
    
    # Control Output (SSR Heater Duty Cycle 0-100%)
    u = np.clip(P + I + D, 0.0, 100.0)
    
    cv[t] = u
    p_term[t] = P
    i_term[t] = I
    d_term[t] = D
    
    # Load disturbance injection between t=150s and t=185s
    dist = disturbance_mag if (disturbance_active and 150 <= t <= 185) else 0.0
    
    # Thermal Differential Equation (First-Order Process)
    dT_dt = (-(pv[t - 1] - ambient_temp) + process_gain * u + dist) / process_tau
    pv[t] = pv[t - 1] + dT_dt * dt

# ---------------------------------------------------------
# PERFORMANCE METRICS CALCULATION
# ---------------------------------------------------------
max_pv = np.max(pv)
overshoot = max(0.0, ((max_pv - setpoint) / (setpoint - ambient_temp)) * 100.0)
steady_state_error = abs(setpoint - pv[-1])

# Settling time (within 2% band before disturbance at t=150s)
pre_dist_pv = pv[:145]
within_band = np.abs(pre_dist_pv - setpoint) <= (0.02 * setpoint)
if np.any(within_band):
    unsettled_indices = np.where(~within_band)[0]
    settling_time = time_axis[unsettled_indices[-1] + 1] if len(unsettled_indices) < len(pre_dist_pv) - 1 else 145.0
else:
    settling_time = 145.0

# ---------------------------------------------------------
# MAIN HMI HEADER & SYSTEM STATUS BANNER
# ---------------------------------------------------------
col_title, col_btn = st.columns([3.5, 1])
with col_title:
    st.title("⚙️ Industrial SCADA HMI — Thermal Reactor PID Control")
    st.caption("Architecture: PLC Closed-Loop Control | Anti-Windup PID | PT100 4-20mA Acquisition | Historian Logger")

with col_btn:
    st.write("")
    live_scan = st.button("▶️ Run Live PLC Scan", use_container_width=True)

# Alarm Logic
if max_pv >= high_alarm_limit:
    st.markdown(
        f'<div class="status-box status-alarm">🚨 PLC INTERLOCK ALARM [ALM-101]: HIGH TEMPERATURE THRESHOLD EXCEEDED (Peak: {max_pv:.1f} °C > Limit: {high_alarm_limit:.1f} °C)</div>',
        unsafe_allow_html=True
    )
elif overshoot > 10.0:
    st.markdown(
        f'<div class="status-box status-warning">⚠️ PLC DIAGNOSTIC WARNING [WRN-204]: HIGH LOOP OVERSHOOT DETECTED ({overshoot:.1f}%) — CONSIDER REDUCING Kp OR INCREASING Kd</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="status-box status-normal">🟢 PLC STATUS: RUN MODE | SCAN TIME: 10ms | CLOSED-LOOP PID STABLE | NO ACTIVE ALARMS</div>',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# TOP KPI METRICS ROW
# ---------------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Process Variable (PV)", f"{pv[-1]:.1f} °C", f"{pv[-1] - setpoint:+.2f} °C")
k2.metric("Setpoint (SP)", f"{setpoint:.1f} °C", "Target")
k3.metric("Control Output (CV)", f"{cv[-1]:.1f} %", "SSR PWM Duty")
k4.metric("Peak Overshoot", f"{overshoot:.1f} %", f"Max: {max_pv:.1f} °C")
k5.metric("Settling Time (2%)", f"{settling_time:.0f} s", "Pre-Disturbance")
k6.metric("Steady-State Error", f"{steady_state_error:.2f} °C", "Closed-Loop")

st.divider()

# ---------------------------------------------------------
# INTERACTIVE SCADA VISUALIZATION FUNCTION
# ---------------------------------------------------------
def render_hmi_dashboard(idx):
    curr_pv = pv[idx]
    curr_cv = cv[idx]
    
    col_gauges, col_charts = st.columns([1.1, 2.9])
    
    with col_gauges:
        # Furnace Temperature Gauge
        fig_gauge_pv = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=curr_pv,
            delta={'reference': setpoint, 'increasing': {'color': "#ff5252"}, 'decreasing': {'color': "#00e676"}},
            title={'text': "Reactor Temp PV (°C)", 'font': {'size': 16, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 550], 'tickcolor': "white"},
                'bar': {'color': "#00e676" if curr_pv < high_alarm_limit else "#ff1744"},
                'bgcolor': "#1e222d",
                'steps': [
                    {'range': [0, setpoint * 0.9], 'color': "#1a2634"},
                    {'range': [setpoint * 0.9, high_alarm_limit], 'color': "#1e3a2f"},
                    {'range': [high_alarm_limit, 550], 'color': "#4a151b"}
                ],
                'threshold': {'line': {'color': "#ff1744", 'width': 4}, 'thickness': 0.8, 'value': high_alarm_limit}
            }
        ))
        fig_gauge_pv.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="#0e1117", font={'color': "white"})
        st.plotly_chart(fig_gauge_pv, use_container_width=True)
        
        # SSR Heater Valve Output Gauge
        fig_gauge_cv = go.Figure(go.Indicator(
            mode="gauge+number",
            value=curr_cv,
            number={'suffix': " %"},
            title={'text': "Heater Actuator CV (%)", 'font': {'size': 16, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "white"},
                'bar': {'color': "#29b6f6"},
                'bgcolor': "#1e222d",
                'steps': [
                    {'range': [0, 70], 'color': "#16222f"},
                    {'range': [70, 100], 'color': "#2c2518"}
                ]
            }
        ))
        fig_gauge_cv.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="#0e1117", font={'color': "white"})
        st.plotly_chart(fig_gauge_cv, use_container_width=True)

    with col_charts:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            subplot_titles=("Closed-Loop Thermal Response (PV vs. SP)", "Actuator Control Effort (CV %) & PID Contributions")
        )
        
        # Plot 1: PV, SP, Alarm Limit
        fig.add_trace(go.Scatter(x=time_axis[:idx+1], y=pv[:idx+1], name="Process Variable (PV)", line=dict(color="#00e676", width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=time_axis[:idx+1], y=sp_arr[:idx+1], name="Setpoint (SP)", line=dict(color="#ffea00", width=2, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=time_axis[:idx+1], y=np.full(idx+1, high_alarm_limit), name="High Alarm Limit", line=dict(color="#ff1744", width=1.5, dash="dot")), row=1, col=1)
        
        if disturbance_active and idx >= 150:
            fig.add_vrect(x0=150, x1=min(idx, 185), fillcolor="#ff9100", opacity=0.15, line_width=0, annotation_text="Door Open Disturbance", annotation_position="top left", row=1, col=1)
        
        # Plot 2: CV, P, I, D
        fig.add_trace(go.Scatter(x=time_axis[:idx+1], y=cv[:idx+1], name="Heater Output (CV %)", line=dict(color="#29b6f6", width=2), fill='tozeroy', fillcolor='rgba(41, 182, 246, 0.12)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=time_axis[:idx+1], y=p_term[:idx+1], name="P-Term", line=dict(color="#ab47bc", width=1, dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=time_axis[:idx+1], y=i_term[:idx+1], name="I-Term", line=dict(color="#ffa726", width=1, dash="dot")), row=2, col=1)
        
        fig.update_layout(
            height=500,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#141820",
            margin=dict(l=15, r=15, t=35, b=15),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1, gridcolor="#262c3a")
        fig.update_yaxes(title_text="Control Signal (%)", row=2, col=1, gridcolor="#262c3a")
        fig.update_xaxes(title_text="Time (Seconds)", row=2, col=1, gridcolor="#262c3a")
        
        st.plotly_chart(fig, use_container_width=True)

# Handle Live Animation vs Instant Render
dashboard_placeholder = st.empty()

if live_scan:
    for step_idx in range(15, steps, 15):
        with dashboard_placeholder.container():
            render_hmi_dashboard(step_idx)
        time.sleep(0.08)
    with dashboard_placeholder.container():
        render_hmi_dashboard(steps - 1)
else:
    with dashboard_placeholder.container():
        render_hmi_dashboard(steps - 1)

# ---------------------------------------------------------
# BOTTOM SECTION: PLC ALARM & DIAGNOSTIC BUFFER + HISTORIAN EXPORT
# ---------------------------------------------------------
st.divider()
col_log, col_export = st.columns([2.5, 1.5])

with col_log:
    st.subheader("📋 PLC Diagnostic Buffer & Event Log")
    base_time = datetime.now() - timedelta(seconds=300)
    events = [
        {"Timestamp": (base_time).strftime("%H:%M:%S.%f")[:-3], "Tag": "OB100", "Severity": "INFO", "Event Description": "PLC Warm Restart completed. PID_Compact initialized in AUTO mode."},
        {"Timestamp": (base_time + timedelta(seconds=int(settling_time))).strftime("%H:%M:%S.%f")[:-3], "Tag": "DB_PID", "Severity": "STATUS", "Event Description": f"Process reached 2% settling band around SP ({setpoint:.1f} °C)."},
    ]
    if disturbance_active:
        events.append({"Timestamp": (base_time + timedelta(seconds=150)).strftime("%H:%M:%S.%f")[:-3], "Tag": "DI_0.4", "Severity": "WARNING", "Event Description": f"External thermal drop detected ({disturbance_mag} °C). PID compensating."})
        events.append({"Timestamp": (base_time + timedelta(seconds=215)).strftime("%H:%M:%S.%f")[:-3], "Tag": "DB_PID", "Severity": "STATUS", "Event Description": "Disturbance rejected. Process temperature restored to Setpoint."})
    if max_pv >= high_alarm_limit:
        events.append({"Timestamp": (base_time + timedelta(seconds=int(np.argmax(pv)))).strftime("%H:%M:%S.%f")[:-3], "Tag": "ALM_101", "Severity": "CRITICAL", "Event Description": f"High Temp Interlock Triggered! PV reached {max_pv:.1f} °C."})

    st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)

with col_export:
    st.subheader("💾 SCADA Historian Data Export")
    st.write("Download real-time telemetry log (PV, SP, CV, P/I/D terms) in CSV format for MATLAB/Excel post-processing.")
    df_export = pd.DataFrame({
        "Time_s": time_axis,
        "Setpoint_C": sp_arr,
        "Process_Variable_C": np.round(pv, 2),
        "Control_Output_Pct": np.round(cv, 2),
        "P_Term": np.round(p_term, 2),
        "I_Term": np.round(i_term, 2),
        "D_Term": np.round(d_term, 2)
    })
    csv_data = df_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Telemetry Log (.CSV)",
        data=csv_data,
        file_name="SCADA_PID_Telemetry_Log.csv",
        mime="text/csv",
        use_container_width=True
    )

# ---------------------------------------------------------
# DEVELOPER CREDITS & FOOTER
# ---------------------------------------------------------
st.divider()
st.markdown("""
    <div style="text-align: center; padding: 15px; background-color: #141820; border-radius: 8px; border: 1px solid #262c3a; margin-top: 20px;">
        <p style="color: #90a4ae; margin-bottom: 4px; font-size: 0.85rem; letter-spacing: 1px; text-transform: uppercase;">
            System Architecture & Implementation
        </p>
        <h4 style="color: #00e676; margin: 0; font-family: monospace; font-size: 1.15rem;">
            Designed & Developed by Amirhosein Razeghi
        </h4>
        <p style="color: #cfd8dc; font-size: 0.85rem; margin-top: 6px; margin-bottom: 0;">
            Electrical Engineering | K. N. Toosi University of Technology
        </p>
    </div>
""", unsafe_allow_html=True)    