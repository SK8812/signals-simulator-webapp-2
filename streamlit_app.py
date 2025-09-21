import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Signal Simulator Workspace", layout="wide")
st.title("SIGNAL SIMULATOR WEB APP")

# ----------------- Workspace -----------------
if "signals" not in st.session_state:
    st.session_state.signals = []

st.sidebar.header("Add a New Signal")
eq = st.sidebar.text_input("Signal Equation (use sin, cos, exp, pi, t)", "sin(2*pi*5*t)")
duration = st.sidebar.slider("Duration (seconds)", 0.5, 20.0, 2.0, step=0.5)
sampling_rate = st.sidebar.slider("Sampling Rate (Hz)", 100, 10000, 2000, step=100)
noise = st.sidebar.slider("Noise Amplitude", 0.0, 1.0, 0.0, step=0.1)
col = st.sidebar.color_picker("Signal Color", "#0000FF")
x_min_input = st.sidebar.number_input("X-axis Min (Optional, leave blank for auto)", value=float("nan"))
x_max_input = st.sidebar.number_input("X-axis Max (Optional, leave blank for auto)", value=float("nan"))
y_min = st.sidebar.number_input("Y-axis Min", value=-2.0)
y_max = st.sidebar.number_input("Y-axis Max", value=2.0)

if st.sidebar.button("Add Signal"):
    st.session_state.signals.append({
        "equation": eq,
        "duration": duration,
        "sampling_rate": sampling_rate,
        "noise": noise,
        "color": col,
        "x_min": x_min_input,
        "x_max": x_max_input,
        "y_min": y_min,
        "y_max": y_max
    })

# ----------------- Display Workspace -----------------
st.subheader("🗂 Workspace: Signals")
to_delete = []
for i, sig in enumerate(st.session_state.signals):
    st.markdown(f"**Signal {i+1}:** {sig['equation']}")
    if st.button(f"Delete Signal {i+1}", key=f"del_{i}"):
        to_delete.append(i)

# Delete selected signals
for i in sorted(to_delete, reverse=True):
    st.session_state.signals.pop(i)

# ----------------- Plot Each Signal -----------------
for i, sig in enumerate(st.session_state.signals):
    st.markdown(f"### Signal {i+1}: {sig['equation']}")

    # Symmetric time vector
    N = int(sig["sampling_rate"] * sig["duration"])
    if N % 2 == 0:
        t = np.linspace(-sig["duration"]/2, sig["duration"]/2, N+1)
    else:
        t = np.linspace(-sig["duration"]/2, sig["duration"]/2, N)

    allowed = {"t": t, "sin": np.sin, "cos": np.cos, "exp": np.exp, "pi": np.pi}

    try:
        y = eval(sig["equation"], {"__builtins__": None}, allowed)
    except:
        y = np.zeros_like(t)

    if sig["noise"] > 0:
        y += sig["noise"] * np.random.normal(size=len(t))

    # Odd/Even
    even_check = np.allclose(y, y[::-1], atol=1e-5)
    odd_check = np.allclose(y, -y[::-1], atol=1e-5)
    if even_check:
        symmetry = "Even Signal"
    elif odd_check:
        symmetry = "Odd Signal"
    else:
        symmetry = "Neither Odd nor Even"

    # Energy/Power
    energy = np.sum(np.abs(y)**2) / sig["sampling_rate"]
    power = np.mean(np.abs(y)**2)
    signal_type = "Energy Signal" if np.isfinite(energy) and energy < 1e6 else "Power Signal"

    st.write(f"**Symmetry:** {symmetry} | **Energy:** {energy:.4f} | **Power:** {power:.4f} | **Type:** {signal_type}")

    # ----------------- Time Domain Plot -----------------
    fig1, ax1 = plt.subplots()
    ax1.plot(t, y, color=sig["color"])
    # Auto x-axis based on duration if not specified
    x_min = sig["x_min"] if not np.isnan(sig["x_min"]) else t[0]
    x_max = sig["x_max"] if not np.isnan(sig["x_max"]) else t[-1]
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([sig["y_min"], sig["y_max"]])
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Time Domain Signal {i+1}")
    st.pyplot(fig1)

    # ----------------- Frequency Domain Plot -----------------
    N_fft = len(y)
    yf = fft(y)
    xf = fftfreq(N_fft, 1 / sig["sampling_rate"])[:N_fft//2]
    fig2, ax2 = plt.subplots()
    ax2.plot(xf, 2.0 / N_fft * np.abs(yf[:N_fft//2]), color=sig["color"])
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title(f"Frequency Spectrum Signal {i+1}")
    ax2.set_xlim([0, sig["sampling_rate"]/2])
    st.pyplot(fig2)

    # ----------------- 3D Waveform -----------------
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.plot3D(t, y, np.zeros_like(t), color=sig["color"])
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.set_zlabel("Depth")
    ax3.set_title(f"3D Waveform Signal {i+1}")
    ax3.set_xlim([x_min, x_max])
    ax3.set_ylim([sig["y_min"], sig["y_max"]])
    st.pyplot(fig3)
