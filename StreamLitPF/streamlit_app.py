# -*- coding: utf-8 -*-
import streamlit as st
import pickle, os, struct, tempfile, io
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(
    page_title="PulseForge - Sleep Apnea AHI Estimator",
    page_icon="🫁",
    layout="wide"
)

@st.cache_resource
def load_model(pkl_path="pulseforge_model.pkl"):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def read_hea(hea_path):
    with open(hea_path, encoding="utf-8") as f:
        parts = f.readline().strip().split()
    return int(parts[1]), int(parts[2]), int(parts[3]) if len(parts) > 3 else 0

def read_ecg_channel(dat_bytes, hea_path, channel=0, gain=200.0):
    n_sig, fs, _ = read_hea(hea_path)
    raw = np.frombuffer(dat_bytes, dtype="<i2")
    ecg = raw[channel::n_sig].astype(np.float32) / gain
    return ecg, fs

def read_annotation_file(data):
    annotations, sample = [], 0
    i = 0
    while i <= len(data) - 2:
        word    = struct.unpack_from("<H", data, i)[0]; i += 2
        anntype = (word >> 10) & 0x3F
        diff    =  word & 0x3FF
        if anntype == 0 and diff == 0:
            break
        if anntype == 59:
            if i <= len(data) - 2:
                hi = struct.unpack_from("<H", data, i)[0]; i += 2
                sample += (hi << 10) | diff
            continue
        sample += diff
        annotations.append((sample, anntype))
    return annotations

def apn_labels_from_bytes(apn_bytes):
    raw = read_annotation_file(apn_bytes)
    return [1 if c == 8 else 0 for _, c in raw if c in (1, 8)]

def bandpass_ecg(epoch, fs=100, low=0.5, high=40.0):
    nyq  = fs / 2
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, epoch)

def bandpower(sig, fs, fmin, fmax):
    nperseg    = min(512, len(sig))
    freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=nperseg)
    mask       = (freqs >= fmin) & (freqs <= fmax)
    return float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0

def detect_r_peaks(epoch, fs=100):
    try:
        filtered   = bandpass_ecg(epoch, fs)
        diff_sq    = np.diff(filtered) ** 2
        win        = int(0.15 * fs)
        integrated = np.convolve(diff_sq, np.ones(win) / win, mode="same")
        threshold  = 0.30 * np.max(integrated)
        peaks, _   = sp_signal.find_peaks(
            integrated, height=threshold, distance=int(0.35 * fs)
        )
        return peaks, filtered
    except Exception:
        return np.array([]), epoch

def spectral_entropy(sig, fs):
    _, psd = sp_signal.welch(sig, fs=fs, nperseg=min(512, len(sig)))
    p      = psd / (psd.sum() + 1e-10)
    return float(-np.sum(p * np.log2(p + 1e-10)))

def extract_features(epoch, fs=100):
    feat = {}
    feat["mean"]      = float(np.mean(epoch))
    feat["std"]       = float(np.std(epoch))
    feat["variance"]  = float(np.var(epoch))
    feat["skewness"]  = float(skew(epoch))
    feat["kurtosis"]  = float(kurtosis(epoch))
    feat["rms"]       = float(np.sqrt(np.mean(epoch ** 2)))
    feat["iqr"]       = float(np.percentile(epoch, 75) - np.percentile(epoch, 25))
    feat["min"]       = float(np.min(epoch))
    feat["max"]       = float(np.max(epoch))
    feat["peak2peak"] = float(np.max(epoch) - np.min(epoch))
    feat["zcr"]       = float(np.sum(np.diff(np.sign(epoch)) != 0) / len(epoch))
    feat["energy"]    = float(np.sum(epoch ** 2) / len(epoch))
    feat["bp_vlf"]          = bandpower(epoch, fs, 0.003, 0.04)
    feat["bp_lf"]           = bandpower(epoch, fs, 0.04,  0.15)
    feat["bp_hf"]           = bandpower(epoch, fs, 0.15,  0.4)
    feat["bp_ecg"]          = bandpower(epoch, fs, 1.0,   40.0)
    feat["lf_hf_ratio"]     = feat["bp_lf"] / (feat["bp_hf"] + 1e-10)
    feat["spectral_entropy"] = spectral_entropy(epoch, fs)
    peaks, filtered = detect_r_peaks(epoch, fs)
    feat["n_peaks"]    = float(len(peaks))
    feat["heart_rate"] = float(len(peaks))
    if len(peaks) >= 3:
        rr_ms = np.diff(peaks) / fs * 1000
        feat["mean_rr"] = float(np.mean(rr_ms))
        feat["sdnn"]    = float(np.std(rr_ms, ddof=1))
        diff_rr         = np.diff(rr_ms)
        feat["rmssd"]   = float(np.sqrt(np.mean(diff_rr ** 2)))
        feat["nn50"]    = float(np.sum(np.abs(diff_rr) > 50))
        feat["pnn50"]   = float(feat["nn50"] / len(diff_rr)) if len(diff_rr) else 0.0
        sd1             = float(np.std(diff_rr / np.sqrt(2), ddof=1))
        feat["sd1"]     = sd1
        feat["sd2"]     = float(np.sqrt(max(2 * feat["sdnn"] ** 2 - 0.5 * sd1 ** 2, 0)))
        feat["sd_ratio"]= feat["sd2"] / (sd1 + 1e-10)
        feat["cv_rr"]   = feat["sdnn"] / (feat["mean_rr"] + 1e-10)
        amps               = filtered[peaks]
        feat["r_amp_mean"] = float(np.mean(amps))
        feat["r_amp_std"]  = float(np.std(amps))
        feat["r_amp_cv"]   = float(np.std(amps) / (np.mean(np.abs(amps)) + 1e-10))
    else:
        for k in ["mean_rr","sdnn","rmssd","nn50","pnn50","sd1","sd2",
                  "sd_ratio","cv_rr","r_amp_mean","r_amp_std","r_amp_cv"]:
            feat[k] = 0.0
    try:
        fe = bandpass_ecg(epoch, fs)
        feat["filt_std"]      = float(np.std(fe))
        feat["filt_kurtosis"] = float(kurtosis(fe))
        feat["filt_rms"]      = float(np.sqrt(np.mean(fe ** 2)))
        feat["filt_iqr"]      = float(np.percentile(fe, 75) - np.percentile(fe, 25))
        feat["filt_entropy"]  = spectral_entropy(fe, fs)
    except Exception:
        for k in ["filt_std","filt_kurtosis","filt_rms","filt_iqr","filt_entropy"]:
            feat[k] = 0.0
    feat["sample_entropy"] = 0.0
    if len(peaks) >= 8:
        try:
            rr_ms     = np.diff(peaks) / fs * 1000
            rr_t      = peaks[1:] / fs
            rr_it     = np.arange(rr_t[0], rr_t[-1], 0.25)
            rr_interp = np.interp(rr_it, rr_t, rr_ms)
            feat["rr_bp_vlf"]  = bandpower(rr_interp, 4, 0.003, 0.04)
            feat["rr_bp_lf"]   = bandpower(rr_interp, 4, 0.04,  0.15)
            feat["rr_bp_hf"]   = bandpower(rr_interp, 4, 0.15,  0.4)
            feat["rr_lf_hf"]   = feat["rr_bp_lf"] / (feat["rr_bp_hf"] + 1e-10)
            feat["rr_total_p"] = feat["rr_bp_vlf"] + feat["rr_bp_lf"] + feat["rr_bp_hf"]
            feat["rr_hf_pct"]  = feat["rr_bp_hf"] / (feat["rr_total_p"] + 1e-10)
        except Exception:
            for k in ["rr_bp_vlf","rr_bp_lf","rr_bp_hf","rr_lf_hf","rr_total_p","rr_hf_pct"]:
                feat[k] = 0.0
    else:
        for k in ["rr_bp_vlf","rr_bp_lf","rr_bp_hf","rr_lf_hf","rr_total_p","rr_hf_pct"]:
            feat[k] = 0.0
    return feat

LAG_FEATURES_DASH = [
    "std","rmssd","sdnn","sd1","sd2","bp_hf",
    "rr_bp_hf","rr_lf_hf","spectral_entropy",
    "peak2peak","heart_rate","r_amp_std"
]

def run_inference(ecg, fs, models, all_features, threshold, epoch_samp=6000):
    n_epochs  = len(ecg) // epoch_samp
    feat_rows = []
    progress  = st.progress(0, text="Extracting features...")
    for i in range(n_epochs):
        ep = ecg[i * epoch_samp : (i + 1) * epoch_samp]
        if len(ep) < epoch_samp:
            continue
        feat_rows.append(extract_features(ep, fs))
        if i % 20 == 0:
            progress.progress(int(i / n_epochs * 80), text=f"Epoch {i}/{n_epochs}...")
    progress.progress(80, text="Running model inference...")
    df = pd.DataFrame(feat_rows)
    for feat in LAG_FEATURES_DASH:
        if feat not in df.columns:
            continue
        for lag in [-2, -1, 1, 2]:
            df[f"{feat}_lag{lag:+d}"] = df[feat].shift(-lag).bfill().ffill()
        df[f"{feat}_roll5"] = df[feat].rolling(5, center=True, min_periods=1).mean()
    for col in all_features:
        if col not in df.columns:
            df[col] = 0.0
    X     = df[all_features].values
    probs = np.stack([m.predict_proba(X)[:, 1] for m in models]).mean(axis=0)
    preds = (probs >= threshold).astype(int)
    progress.progress(100, text="Done!")
    return probs, preds, df

def ahi_severity(ahi):
    if ahi < 5:  return "Normal",   "#d4edda"
    if ahi < 15: return "Mild",     "#fff3cd"
    if ahi < 30: return "Moderate", "#ffe0b2"
    return "Severe", "#f8d7da"

# ── Timeline plot → bytes buffer (reusable for PDF + screen) ─────────────────
def make_timeline_buf(probs, preds, threshold, ahi, sev, gt_labels=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    ax1.plot(probs, color="steelblue", lw=1.2, label="Apnea probability")
    ax1.axhline(threshold, color="tomato", ls="--", lw=1,
                label=f"Threshold = {threshold:.2f}")
    ax1.fill_between(range(len(probs)), probs, threshold,
                     where=(probs >= threshold), alpha=0.3, color="tomato",
                     label="Apnea region")
    if gt_labels:
        n = min(len(gt_labels), len(probs))
        ax1.plot(range(n), gt_labels[:n], "g-", alpha=0.5, lw=1.5,
                 label="Ground truth")
    ax1.set_ylabel("Apnea Probability")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8)
    ax1.set_title(f"Epoch Timeline  |  AHI = {ahi}  |  {sev}")
    ax2.bar(range(len(preds)), preds,
            color=["tomato" if p else "steelblue" for p in preds], width=1.0)
    ax2.set_ylabel("Apnea (1/0)")
    ax2.set_xlabel("Epoch (minutes)")
    ax2.set_yticks([0, 1])
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

# ── PDF report generation (fpdf2, 1-page, Colab notebook style) ──────────────
def generate_pdf_report(record_name, ahi, sev, hours, n_apnea, n_total,
                        model_auc, model_f1, threshold, timeline_buf,
                        rec_auc=None, rec_f1=None, gt_ahi=None):
    from fpdf import FPDF, XPos, YPos

    # fpdf2 needs a file path for images — write buf to a temp PNG
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png", mode="wb") as tmp:
        tmp.write(timeline_buf.read())
        img_path = tmp.name

    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # ── Header bar (blue, white text) ─────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_fill_color(30, 80, 160)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, "Sleep Apnea AHI Estimation Report",
             fill=True, align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 5,
             "DISCLAIMER: This report is generated by an automated ML pipeline "
             "for research purposes only. Not a medical diagnosis.",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(3)

    # ── Record info ───────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, f"Record: {record_name}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Recording Duration  : {hours} hours",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Apnea Epochs Detected : {n_apnea} / {n_total}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # ── AHI coloured box ──────────────────────────────────────────────────────
    colour_map = {
        "Normal"  : (0,   180,   0),
        "Mild"    : (230, 140,   0),
        "Moderate": (210,  80,   0),
        "Severe"  : (200,   0,   0),
    }
    r, g, b = colour_map.get(sev, (100, 100, 100))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 14, f"AHI = {ahi}   |   Severity: {sev}",
             fill=True, align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # ── Model performance ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Model Performance (GroupKFold OOF - no data leakage)",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6,
             f"ROC-AUC : {model_auc:.4f}   |   F1-Score : {model_f1:.4f}   |   "
             f"Threshold : {threshold:.4f}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Optional per-record validation row
    if rec_auc is not None and rec_f1 is not None:
        pdf.set_font("Helvetica", "I", 9)
        gt_str = f"  |  GT AHI : {gt_ahi}" if gt_ahi is not None else ""
        pdf.cell(0, 5,
                 f"Record-level  AUC : {rec_auc:.4f}   |   "
                 f"F1 : {rec_f1:.4f}{gt_str}",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # ── Epoch timeline plot ───────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, "Epoch-level Apnea Timeline",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.image(img_path, w=180)
    pdf.ln(3)

    # ── AHI severity scale ────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 5, "AHI Severity Scale",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5,
             "< 5 : Normal   |   5-15 : Mild   |   15-30 : Moderate   |   >= 30 : Severe",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5,
             "Generated by PulseForge | YUGUO Hackathon 2026 | "
             "SSN College of Engineering | Team PulseForge",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    pdf_bytes = bytes(pdf.output())

    try:
        os.unlink(img_path)
    except Exception:
        pass

    return pdf_bytes


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Configuration")
    model_path = st.text_input("Model pickle path", "pulseforge_model.pkl")
    st.markdown("---")
    st.markdown("**Expected files per record:**")
    st.markdown("- `.dat` - ECG signal")
    st.markdown("- `.hea` - Header file")
    st.markdown("- `.apn` - Annotations (optional)")
    st.markdown("---")
    st.markdown("**AHI Severity Scale**")
    st.markdown("- < 5  : Normal")
    st.markdown("- 5-15 : Mild")
    st.markdown("- 15-30: Moderate")
    st.markdown("- >= 30: Severe")

st.title("PulseForge - Sleep Apnea AHI Estimator")
st.caption("YUGUO Hackathon 2026 | Team PulseForge | SSN College of Engineering")
st.warning(
    "DISCLAIMER: This tool is for research purposes only and does not "
    "constitute a medical diagnosis. Always consult a qualified physician."
)

try:
    artifact      = load_model(model_path)
    models_loaded = artifact["models"]
    all_features  = artifact["all_features"]
    threshold     = artifact["threshold"]
    st.sidebar.success(
        f"Model loaded\nAUC = {artifact['oof_auc']:.4f} | "
        f"F1 = {artifact['oof_f1']:.4f}"
    )
except Exception as e:
    st.error(f"Could not load model from '{model_path}': {e}")
    st.info("Make sure pulseforge_model.pkl is in the same folder as this app.")
    st.stop()

st.markdown("---")
st.header("Upload Record Files")

col1, col2, col3 = st.columns(3)
with col1:
    dat_file = st.file_uploader("ECG signal (.dat)", type=["dat"])
with col2:
    hea_file = st.file_uploader("Header (.hea)", type=["hea"])
with col3:
    apn_file = st.file_uploader("Annotations (.apn) - optional", type=["apn"])

if dat_file and hea_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".hea", mode="wb") as tmp:
        tmp.write(hea_file.read())
        hea_tmp = tmp.name

    run_btn = st.button("Run AHI Estimation", type="primary", use_container_width=True)

    if run_btn:
        ecg, fs = read_ecg_channel(dat_file.read(), hea_tmp)
        probs, preds, feat_df = run_inference(
            ecg, fs, models_loaded, all_features, threshold
        )

        n_apnea = int(preds.sum())
        n_total = len(preds)
        ahi     = round((n_apnea / n_total) * 60, 1)
        sev, bg = ahi_severity(ahi)
        hours   = round(n_total / 60, 2)

        gt_labels = None
        if apn_file:
            gt_labels = apn_labels_from_bytes(apn_file.read())

        # Metrics row
        st.markdown("---")
        st.header("Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AHI Score",          f"{ahi}")
        m2.metric("Severity",           sev)
        m3.metric("Recording Duration", f"{hours} hrs")
        m4.metric("Apnea Epochs",       f"{n_apnea} / {n_total}")

        st.markdown(
            f'<div style="background:{bg};padding:16px;border-radius:8px;'
            f'text-align:center;font-size:1.4em;font-weight:bold;">'
            f'AHI = {ahi} &nbsp;|&nbsp; {sev}</div>',
            unsafe_allow_html=True
        )

        # Timeline
        st.markdown("---")
        st.subheader("Epoch-level Apnea Timeline")
        timeline_buf = make_timeline_buf(probs, preds, threshold, ahi, sev, gt_labels)
        st.image(timeline_buf, use_container_width=True)
        timeline_buf.seek(0)   # reset for PDF use

        # HRV feature table
        st.markdown("---")
        st.subheader("Per-Epoch Feature Summary (HRV)")
        display_cols = [c for c in
            ["mean_rr","sdnn","rmssd","sd1","sd2","heart_rate","rr_bp_hf","spectral_entropy"]
            if c in feat_df.columns]
        st.dataframe(feat_df[display_cols].describe().T.round(3), use_container_width=True)

        # Ground truth validation
        rec_auc_val = rec_f1_val = gt_ahi_val = None
        if gt_labels and len(gt_labels) >= len(preds):
            from sklearn.metrics import roc_auc_score, f1_score
            gt = np.array(gt_labels[:len(preds)])
            try:
                rec_auc_val = roc_auc_score(gt, probs)
                rec_f1_val  = f1_score(gt, preds)
                gt_ahi_val  = round((int(sum(gt)) / len(gt)) * 60, 1)
                st.markdown("---")
                st.subheader("Validation Against Ground Truth")
                c1, c2 = st.columns(2)
                c1.metric("Record ROC-AUC", f"{rec_auc_val:.4f}")
                c2.metric("Record F1-Score", f"{rec_f1_val:.4f}")
                st.info(
                    f"Ground Truth AHI = {gt_ahi_val}  |  "
                    f"Predicted AHI = {ahi}  |  "
                    f"Difference = {abs(ahi - gt_ahi_val):.1f}"
                )
            except Exception:
                pass

        # PDF download
        st.markdown("---")
        record_name = dat_file.name.replace(".dat", "")
        with st.spinner("Generating PDF report..."):
            pdf_bytes = generate_pdf_report(
                record_name = record_name,
                ahi         = ahi,
                sev         = sev,
                hours       = hours,
                n_apnea     = n_apnea,
                n_total     = n_total,
                model_auc   = artifact["oof_auc"],
                model_f1    = artifact["oof_f1"],
                threshold   = threshold,
                timeline_buf= timeline_buf,
                rec_auc     = rec_auc_val,
                rec_f1      = rec_f1_val,
                gt_ahi      = gt_ahi_val,
            )

        st.download_button(
            label     = "Download PDF Report",
            data      = pdf_bytes,
            file_name = f"ahi_report_{record_name}.pdf",
            mime      = "application/pdf",
            use_container_width=True,
        )

else:
    st.info("Upload a .dat and .hea file above to begin.")
    st.markdown("**How it works:**")
    st.markdown(
        "1. Upload your PhysioNet Apnea-ECG record files\n"
        "2. The ECG is segmented into 60-second epochs\n"
        "3. 42+ features are extracted per epoch (HRV, spectral, statistical)\n"
        "4. LightGBM ensemble predicts apnea probability per epoch\n"
        "5. AHI is computed from predicted apnea epoch count\n"
        "6. Severity is classified using standard clinical thresholds\n"
        "7. A 1-page PDF report is generated for download"
    )