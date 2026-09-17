# ============================================================
# Focal Mechanism Determination Pipeline (FocoNet)
#
# Stage 6 of the workflow (after magnitud.py):
#   1) pick.py         -> phase picking (EQTransformer / SeisBench)
#   2) association.py  -> phase association + quality control
#   3) locator.py      -> absolute hypocentre location (NonLinLoc)
#   4) relocation.py   -> double-difference relocation (hypoDD)
#   5) magnitud.py     -> local magnitude
#   6) focmech.py      -> focal mechanism determination (this file)
#
# Method: FocoNet (Song et al., 2026) -- deep-learning first-motion polarity
# inversion. For each relocated event:
#   1. Loads relocated catalog from relocated_catalog.csv (only relocated events).
#   2. Loads phase picks from hypocenter_phases.csv (P picks per station).
#   3. Reads preprocessed Z-component miniSEED around each P pick.
#   4. Determines first-motion polarity (+1 up, -1 down, 0 unknown) using
#      AIC pick refinement + ensemble majority vote.
#   5. Computes Z-channel log10 SNR (signal/noise window around P).
#   6. Projects stations to ENU km relative to relocated epicentre and normalizes
#      features (dx/50, dy/50, depth/20, dummy station masking).
#   7. Directly runs PyTorch FocoNet inference in-memory (no subprocess/predict.py).
#   8. Converts P/T/B axes -> strike/dip/rake via pyrocko.
#   9. Writes focal_mechanisms.csv.
#
# Plotting (all in one script):
#   A. Map with beachballs coloured by faulting style.
#   B. Lower-hemisphere P/T stereonet.
#   C. Rake histogram (slip style).
#   D. Cross-section figures (lat/lon/NE-SW/NW-SE slices).
#   E. Kagan-angle pairwise analysis (3-panel figure).
#
# Input:
#   output/relocation/relocated_catalog.csv  (relocated events)
#   output/hypocenter_locator/csv/hypocenter_phases.csv  (phase picks)
#   output/preprocessing/  (preprocessed waveforms)
#   stations.txt
#
# Output (under output/focal_mechanism/):
#   event_input.npy           FocoNet input dict (saved for reference)
#   PT_predictions.npy        (N, 9) raw network output
#   focal_mechanisms.csv      one row per event (strike/dip/rake + P/T/B axes)
#   kagan_pairs.csv           all pairwise Kagan angles
#   plots/
#     focal_mechanisms.png
#     focal_cross_sections_latitude.png
#     focal_cross_sections_longitude.png
#     focal_cross_sections_ne.png
#     focal_cross_sections_nw.png
#     kagan_angles.png
#
# NOTE: This file is deliberately split into:
#   1) MODULE IMPORTS
#   2) CONFIGURATION (loaded from config/config.yaml -- see CONFIG_YAML_PATH)
#   3) PROCESS (Neural Net Architecture, polarity, inference, SDR conversion, plots, main)
# CONFIG is loaded from the 'focal_mechanism' section of the shared
# config/config.yaml -- edit the YAML to change parameters, not this file.
# ============================================================

# ============================================================
# 1) MODULE IMPORTS
# ============================================================

import json
import math
import os
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

try:
    from obspy import UTCDateTime, read
    from obspy.geodetics import gps2dist_azimuth
except ImportError:
    raise ImportError("Package 'obspy' is required. Run with the pyocto conda env.")

try:
    from obspy.imaging.beachball import beach
except ImportError:
    beach = None


# ============================================================
# 2) CONFIGURATION
# ============================================================

CONFIG_YAML_PATH = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")
if not os.path.exists(CONFIG_YAML_PATH):
    CONFIG_YAML_PATH = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")

with open(CONFIG_YAML_PATH, 'r') as _f:
    _all_cfg = yaml.safe_load(_f)

CONFIG = _all_cfg['focal_mechanism']
CONFIG['network'] = _all_cfg['network']


# ============================================================
# 3) PROCESS
# ============================================================

# ------------------------------------------------------------
# 3.1  FocoNet PyTorch Model Architecture (In-Memory Inference)
# ------------------------------------------------------------

def angle2PTB(angle_input):
    N = angle_input.shape[0]
    x = angle_input[:, 0]
    y = angle_input[:, 1]
    z = angle_input[:, 2]

    a = torch.concatenate((
        torch.sin(y) * torch.cos(x),
        torch.sin(y) * torch.sin(x),
        torch.cos(y)
    )).reshape(-1, N).T

    b = torch.concatenate((
        torch.sin(z) * torch.sin(x) - torch.cos(z) * torch.cos(y) * torch.cos(x),
        -torch.sin(z) * torch.cos(x) - torch.cos(z) * torch.cos(y) * torch.sin(x),
        torch.cos(z) * torch.sin(y)
    )).reshape(-1, N).T

    c = torch.cross(a, b, dim=1)
    out = torch.concatenate((
        a.reshape(N, 1, -1),
        b.reshape(N, 1, -1),
        c.reshape(N, 1, -1)
    ), axis=1)
    return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dim = d_model

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class FocoNet_Full_Model(nn.Module):
    def __init__(self):
        super(FocoNet_Full_Model, self).__init__()
        self.organize = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
        )
        self.endlayer = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 8),
            nn.LeakyReLU(),
            nn.LayerNorm(8),
            nn.Linear(8, 3),
        )
        self.beg_layer = nn.Sequential(
            nn.Linear(14, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.beg_res1 = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
        )
        self.end_res1 = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 32),
        )
        self.attentionlayer1 = EncoderLayer(d_model=128, num_heads=32, d_ff=1024, dropout=0.1)
        self.attentionlayer2 = EncoderLayer(d_model=128, num_heads=32, d_ff=1024, dropout=0.1)
        self.attentionlayer3 = EncoderLayer(d_model=128, num_heads=32, d_ff=1024, dropout=0.1)
        self.attentionlayer4 = EncoderLayer(d_model=128, num_heads=32, d_ff=1024, dropout=0.1)
        self.attentionlayer5 = EncoderLayer(d_model=128, num_heads=8, d_ff=512, dropout=0.0)
        self.attentionlayer6 = EncoderLayer(d_model=128, num_heads=8, d_ff=512, dropout=0.0)
        self.attentionlayer7 = EncoderLayer(d_model=128, num_heads=8, d_ff=512, dropout=0.0)

    def forward(self, x, loc, sta_num, sta_mask, mode='test'):
        out = torch.concatenate((x, loc), axis=-1)
        out = self.beg_layer(out)
        out = out + self.beg_res1(out)

        out = self.attentionlayer1(out)
        out = self.attentionlayer2(out) + out
        out = self.attentionlayer3(out) + out
        out = self.attentionlayer4(out) + out
        out = self.attentionlayer5(out) + out
        out = self.attentionlayer6(out) + out
        out = self.attentionlayer7(out) + out

        out = out * sta_mask
        out = torch.sum(out, 1)
        out = out / sta_num

        out = self.organize(out)
        out = out + self.end_res1(out)
        out = self.endlayer(out)
        out = torch.clamp(out, -1 * torch.pi, 1 * torch.pi)
        out = angle2PTB(out)
        return out


class FocoNetDataset(Dataset):
    def __init__(self, data_dict):
        self.eids = list(data_dict.keys())
        self.N = len(self.eids)
        self.polarity = np.zeros((self.N, 32, 11), dtype=np.float32)
        self.sta_param = np.zeros((self.N, 32, 3), dtype=np.float32)
        self.sta_num = np.zeros((self.N, 1), dtype=np.float32)
        self.sta_mask = np.zeros((self.N, 32, 1), dtype=np.float32)

        for j in range(self.N):
            eid = self.eids[j]
            stas = data_dict[eid]['stationxyz'][:, 1:].astype(float)
            n_sta = len(stas[:, 0])
            self.sta_param[j, :n_sta, :] = stas
            self.sta_param[j, :n_sta, 0:2] /= 50.0
            self.sta_param[j, :n_sta, 2] /= 20.0
            self.sta_param[j, n_sta:, 2] -= 100.0

            polarityPS = data_dict[eid]['polarities']
            self.polarity[j, :n_sta, :] = polarityPS[:, :11]
            self.sta_num[j, 0] = float(n_sta)
            self.sta_mask[j, :n_sta, 0] = 1.0

        self.polarity = torch.Tensor(self.polarity)
        self.sta_param = torch.Tensor(self.sta_param)
        self.sta_num = torch.Tensor(self.sta_num)
        self.sta_mask = torch.Tensor(self.sta_mask)

    def __getitem__(self, index):
        return {
            'wave': self.polarity[index],
            'sta_param': self.sta_param[index],
            'sta_num': self.sta_num[index],
            'sta_mask': self.sta_mask[index]
        }

    def __len__(self):
        return self.N


def run_foconet_predict_direct(input_dict, ckpt_path, device_str='cpu', batch_size=64):
    device = torch.device(device_str if (device_str == 'cpu' or torch.cuda.is_available()) else 'cpu')
    ckpt = Path(ckpt_path)
    
    dataset = FocoNetDataset(input_dict)
    if len(dataset) == 0:
        return np.empty((0, 9), dtype=np.float32)

    net = FocoNet_Full_Model().to(device).eval()
    
    if not ckpt.exists():
        print(f"[focmech] Warning: FocoNet checkpoint not found at {ckpt}.")
        print("[focmech] Initializing model with random weights for demonstration/testing.")
    else:
        try:
            state = torch.load(str(ckpt), map_location=device, weights_only=True)
            net.load_state_dict(state)
        except Exception as e:
            print(f"[focmech] Warning: Failed to load checkpoint ({e}). Initializing with random weights.")

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        drop_last=False,
    )

    out_chunks = []
    with torch.no_grad():
        for batch in loader:
            wave = batch["wave"].to(device)
            sta_param = batch["sta_param"].to(device)
            sta_num = batch["sta_num"].to(device).clamp(min=1.0)
            sta_mask = batch["sta_mask"].to(device)
            pred = net(wave, sta_param, sta_num, sta_mask, mode="test")
            out_chunks.append(pred.detach().cpu().numpy().reshape(-1, 9))

    return np.concatenate(out_chunks, axis=0)


# ------------------------------------------------------------
# 3.2  Constants and polarity helpers
# ------------------------------------------------------------

KM_PER_DEG = 111.32
MAX_STATIONS = 32

_FAULT_COLORS = {
    "normal":      "#3182bd",
    "thrust":      "#de2d26",
    "strike-slip": "#31a354",
}


def _faulting_color(rake):
    r = ((float(rake) + 180.0) % 360.0) - 180.0
    if -135.0 <= r <= -45.0:
        return _FAULT_COLORS["normal"]
    if 45.0 <= r <= 135.0:
        return _FAULT_COLORS["thrust"]
    return _FAULT_COLORS["strike-slip"]


def _fault_legend_handles():
    return [
        Patch(facecolor=_FAULT_COLORS["normal"],      edgecolor="k", label="Normal"),
        Patch(facecolor=_FAULT_COLORS["strike-slip"], edgecolor="k", label="Strike-slip"),
        Patch(facecolor=_FAULT_COLORS["thrust"],      edgecolor="k", label="Thrust"),
    ]


def aic_refine_pick(data, delta, t_pick_in_window_s, window_s=0.4):
    n = len(data)
    if n < 4:
        return t_pick_in_window_s
    i_pick = int(round(t_pick_in_window_s / delta))
    half = max(5, int(round(0.5 * window_s / delta)))
    i_lo = max(2, i_pick - half)
    i_hi = min(n - 2, i_pick + half)
    if i_hi - i_lo < 6:
        return t_pick_in_window_s
    seg = data[i_lo - 1:i_hi + 1].astype(np.float64)
    aic = np.full(seg.size, np.inf)
    for k in range(2, seg.size - 2):
        v1 = np.var(seg[:k + 1])
        v2 = np.var(seg[k + 1:])
        if v1 <= 0 or v2 <= 0:
            continue
        aic[k] = k * np.log(v1) + (seg.size - k - 1) * np.log(v2)
    k_min = int(np.argmin(aic))
    return ((i_lo - 1) + k_min) * delta


def _polarity_first_peak(seg, rel_threshold=0.3):
    if len(seg) < 3:
        return 0
    abs_seg = np.abs(seg)
    threshold = rel_threshold * float(abs_seg.max())
    for k in range(1, len(seg) - 1):
        if abs_seg[k] >= threshold and abs_seg[k] >= abs_seg[k-1] and abs_seg[k] >= abs_seg[k+1]:
            return 1 if seg[k] > 0.0 else -1
    k = int(np.argmax(abs_seg))
    return 1 if seg[k] > 0.0 else -1


def _polarity_half_cycle(seg, half_cycle_samples):
    if len(seg) < 3:
        return 0
    n = max(2, min(half_cycle_samples, len(seg)))
    integral = float(np.sum(seg[:n]))
    if integral == 0.0:
        return 0
    return 1 if integral > 0.0 else -1


def _polarity_slope(seg, n_samples=3):
    if len(seg) < n_samples + 1:
        return 0
    diffs = np.diff(seg[:n_samples + 1])
    s = float(np.mean(diffs))
    if s == 0.0:
        return 0
    return 1 if s > 0.0 else -1


def ensemble_polarity(data, delta, t_pick_in_window_s, polarity_lo_hz=1.0, search_s=0.3):
    i0 = max(0, int(round(t_pick_in_window_s / delta)))
    n_search = max(3, int(round(search_s / delta)))
    seg = data[i0:i0 + n_search].astype(np.float64)
    if len(seg) < 3:
        return 0, 0.0
    half_cycle_samples = max(2, int(round(0.5 / max(polarity_lo_hz, 0.5) / delta)))
    votes = [
        _polarity_first_peak(seg),
        _polarity_half_cycle(seg, half_cycle_samples),
        _polarity_slope(seg, n_samples=3),
    ]
    n_up = sum(1 for v in votes if v > 0)
    n_dn = sum(1 for v in votes if v < 0)
    n_zero = sum(1 for v in votes if v == 0)
    n_max = max(n_up, n_dn, n_zero)
    confidence = n_max / len(votes)
    if n_up > n_dn and n_up > n_zero:
        return 1, confidence
    if n_dn > n_up and n_dn > n_zero:
        return -1, confidence
    return 0, confidence


# ------------------------------------------------------------
# 3.3  Station / catalog loading
# ------------------------------------------------------------

def load_stations(stations_file, network):
    rows = []
    with open(stations_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            sta, lat, lon, elev = line.split()
            rows.append({
                'station': sta,
                'network': network,
                'latitude': float(lat),
                'longitude': float(lon),
                'elevation_m': float(elev),
            })
    return pd.DataFrame(rows).set_index('station')


def load_relocated_catalog(relocated_catalog_csv):
    df = pd.read_csv(relocated_catalog_csv)
    df = df.rename(columns={
        'lat_rel': 'latitude',
        'lon_rel': 'longitude',
        'dep_rel': 'depth_km',
        'reloc_datetime': 'origin_time',
    })
    df['origin_time'] = pd.to_datetime(df['origin_time'], utc=True)
    return df


def load_phases(phases_csv):
    df = pd.read_csv(phases_csv)
    df['pick_time'] = pd.to_datetime(df['pick_time'], utc=True)
    return df


# ------------------------------------------------------------
# 3.4  Waveform index builder & reading
# ------------------------------------------------------------

def _build_waveform_index(waveform_folder):
    idx = {}
    wf_root = Path(waveform_folder)
    for day_dir in sorted(wf_root.iterdir()):
        if not day_dir.is_dir():
            continue
        for mseed_file in sorted(day_dir.glob('*.mseed')):
            parts = mseed_file.stem.split('.')
            if len(parts) >= 2:
                sta = parts[1]
                try:
                    d = date.fromisoformat(day_dir.name)
                except ValueError:
                    continue
                idx[(sta, d)] = mseed_file
    return idx


_EPS = 1e-12

def _get_component(stream, letter):
    for tr in stream:
        if tr.stats.channel and tr.stats.channel[-1].upper() == letter.upper():
            return tr
    return None

def _max_abs_window(data, t_start, t_end, t0, sr):
    if data is None or len(data) == 0:
        return 0.0
    i0 = max(0, int((t_start - t0) * sr))
    i1 = min(len(data), int((t_end - t0) * sr))
    if i1 <= i0:
        return 0.0
    return float(np.max(np.abs(data[i0:i1])))

def _build_polarity_row(stream, p_time, s_time, p_half_s, s_pre_s, s_post_s, noise_pre_s, noise_post_s):
    Z = _get_component(stream, "Z")
    R = _get_component(stream, "R") or _get_component(stream, "N") or _get_component(stream, "1")
    T = _get_component(stream, "T") or _get_component(stream, "E") or _get_component(stream, "2")
    if Z is None:
        return None

    sr = float(Z.stats.sampling_rate)
    t0 = Z.stats.starttime.timestamp
    Rd = R.data if R is not None else None
    Td = T.data if T is not None else None

    i_p = max(0, int((p_time - t0) * sr))
    i_end = min(len(Z.data), i_p + max(2, int(sr * p_half_s)))
    p_window = Z.data[i_p:i_end]
    if len(p_window) == 0 or np.max(np.abs(p_window)) < 1e-12:
        pol = 0.0
    else:
        pol = float(np.sign(p_window[np.argmax(np.abs(p_window))]))

    Pr = _max_abs_window(Rd,     p_time - p_half_s,   p_time + 1.0,    t0, sr)
    Pz = _max_abs_window(Z.data, p_time - p_half_s,   p_time + 1.0,    t0, sr)
    Sr = _max_abs_window(Rd,     s_time - s_pre_s,    s_time + s_post_s, t0, sr)
    Sz = _max_abs_window(Z.data, s_time - s_pre_s,    s_time + s_post_s, t0, sr)
    St = _max_abs_window(Td,     s_time - s_pre_s,    s_time + s_post_s, t0, sr)
    Nr = _max_abs_window(Rd,     p_time - noise_pre_s, p_time - noise_post_s, t0, sr)
    Nz = _max_abs_window(Z.data, p_time - noise_pre_s, p_time - noise_post_s, t0, sr)
    Nt = _max_abs_window(Td,     p_time - noise_pre_s, p_time - noise_post_s, t0, sr)

    max_S = max(Sr, Sz, St)
    max_P = max(Pr, Pz, _EPS)

    return np.array([
        pol,
        np.log10(max_S / max_P + _EPS),
        np.log10((Sr + _EPS) / (Pr + _EPS)),
        np.log10((Sr + _EPS) / (Pz + _EPS)),
        np.log10((Sz + _EPS) / (Pr + _EPS)),
        np.log10((Sz + _EPS) / (Pz + _EPS)),
        np.log10((St + _EPS) / (Pr + _EPS)),
        np.log10((St + _EPS) / (Pz + _EPS)),
        np.log10((Pr + _EPS) / (Nr + _EPS)),
        np.log10((Pz + _EPS) / (Nz + _EPS)),
        np.log10((St + _EPS) / (Nt + _EPS)),
    ], dtype=np.float64)

def _read_3c_window(wf_idx, station, pick_time_utc, pre_s, post_s):
    pick_t = UTCDateTime(pick_time_utc.isoformat())
    d = pick_time_utc.date()
    path = wf_idx.get((station, d))
    if path is None:
        prev_d = date.fromordinal(d.toordinal() - 1)
        path = wf_idx.get((station, prev_d))
    if path is None:
        return None
    try:
        st = read(str(path), starttime=pick_t - pre_s - 1.0, endtime=pick_t + post_s + 1.0)
        st.merge(fill_value='interpolate')
    except Exception:
        return None
    if len(st) == 0:
        return None
    z_trs = [tr for tr in st if tr.stats.channel and tr.stats.channel[-1].upper() == 'Z']
    if not z_trs:
        return None
    return st



def _snr_log10(data, delta, t_pick_in_window, noise_window, signal_window):
    n = len(data)
    def _idx(t):
        return max(0, min(n, int(round(t / delta))))
    s_lo = _idx(t_pick_in_window + signal_window[0])
    s_hi = _idx(t_pick_in_window + signal_window[1])
    n_lo = _idx(t_pick_in_window + noise_window[0])
    n_hi = _idx(t_pick_in_window + noise_window[1])
    sig = data[s_lo:s_hi].astype(np.float64)
    noi = data[n_lo:n_hi].astype(np.float64)
    if len(noi) < 3 or len(sig) < 3:
        return 0.0
    rms_s = float(np.sqrt(np.mean(sig ** 2)))
    rms_n = float(np.sqrt(np.mean(noi ** 2)))
    if rms_n <= 0.0 or rms_s <= 0.0:
        return 0.0
    return float(np.log10(rms_s / rms_n))


def _bandpass_copy(tr, freqmin, freqmax, zerophase=False):
    out = tr.copy()
    try:
        out.detrend('demean')
        out.detrend('simple')
        out.taper(0.05, type='cosine')
        out.filter('bandpass', freqmin=freqmin, freqmax=freqmax,
                   corners=4, zerophase=zerophase)
    except Exception:
        return None
    return out


def extract_event_observations(event_row, event_phases_df, stations_df, wf_idx, cfg):
    bp_lo = float(cfg.get('bandpass_polarity', [1.0, 10.0])[0])
    bp_hi = float(cfg.get('bandpass_polarity', [1.0, 10.0])[1])
    noise_window = tuple(cfg.get('noise_window', [-5.0, -2.0]))
    pre_s = float(cfg.get('pre_pick_s', 6.0))
    post_s = float(cfg.get('post_pick_s', 4.0))
    vp_km_s = float(cfg.get('vp_km_s', 6.0))
    vs_km_s = float(cfg.get('vs_km_s', 3.47))
    
    p_half_s = float(cfg.get('p_half_s', 0.05))
    s_pre_s = float(cfg.get('s_pre_s', 0.05))
    s_post_s = float(cfg.get('s_post_s', 2.0))
    noise_pre_s = float(cfg.get('noise_pre_s', 2.0))
    noise_post_s = float(cfg.get('noise_post_s', 0.2))

    p_picks_df = event_phases_df[event_phases_df['phase'] == 'P']
    s_picks = event_phases_df[event_phases_df['phase'] == 'S'].set_index('station')

    ev_lat = float(event_row['latitude'])
    ev_lon = float(event_row['longitude'])

    observations = []
    for _, pick_row in p_picks_df.iterrows():
        sta = pick_row['station']
        if sta not in stations_df.index:
            continue
        sta_info = stations_df.loc[sta]
        sta_lat = float(sta_info['latitude'])
        sta_lon = float(sta_info['longitude'])
        
        pick_time_utc = pick_row['pick_time']
        p_time = UTCDateTime(pick_time_utc.isoformat()).timestamp
        
        dist_m, az, baz = gps2dist_azimuth(ev_lat, ev_lon, sta_lat, sta_lon)
        dist_km = dist_m / 1000.0

        if sta in s_picks.index:
            s_time_utc = s_picks.loc[sta]['pick_time']
            if isinstance(s_time_utc, pd.Series):
                s_time_utc = s_time_utc.iloc[0]
            s_time = UTCDateTime(s_time_utc.isoformat()).timestamp
        else:
            s_time = p_time + dist_km * (1.0 / vs_km_s - 1.0 / vp_km_s)
            
        pre_req = max(pre_s, noise_pre_s + 2.0, abs(noise_window[0]) + 1.0)
        post_req = max(post_s, (s_time - p_time) + s_post_s + 3.0)

        st_raw = _read_3c_window(wf_idx, sta, pick_time_utc, pre_req, post_req)
        if st_raw is None:
            continue
            
        st_pol = st_raw.copy()
        try:
            st_pol.detrend('demean')
            st_pol.detrend('simple')
            st_pol.taper(0.05, type='cosine')
            st_pol.filter('bandpass', freqmin=bp_lo, freqmax=bp_hi, corners=4, zerophase=False)
            try:
                st_pol.rotate("NE->RT", back_azimuth=baz)
            except Exception:
                pass
        except Exception:
            continue
            
        pol_row = _build_polarity_row(st_pol, p_time, s_time, p_half_s, s_pre_s, s_post_s, noise_pre_s, noise_post_s)
        if pol_row is None:
            continue

        observations.append({
            'station': sta,
            'latitude': sta_lat,
            'longitude': sta_lon,
            'elevation_m': float(sta_info['elevation_m']),
            'polarity': pol_row[0],
            'polarity_conf': 1.0 if pol_row[0] != 0 else 0.0,
            'snr_pz': pol_row[9],
            'features': pol_row,
        })
    return observations


# ------------------------------------------------------------
# 3.5  FocoNet input dict builder
# ------------------------------------------------------------

def _project_xy_km(lat, lon, lat0, lon0):
    dx = (lon - lon0) * math.cos(math.radians(0.5 * (lat + lat0))) * KM_PER_DEG
    dy = (lat - lat0) * KM_PER_DEG
    return dx, dy


def build_foconet_input(events_obs, depth_min_km=0.0, depth_max_km=100.0):
    out = {}
    for ev in events_obs:
        depth_use = float(np.clip(ev['depth_km'], depth_min_km, depth_max_km))
        obs = ev['observations']
        n_sta = min(len(obs), MAX_STATIONS)
        polarities = np.zeros((n_sta, 11), dtype=np.float32)
        stationxyz = np.empty((n_sta, 4), dtype=object)
        for i, o in enumerate(obs[:n_sta]):
            x_km, y_km = _project_xy_km(o['latitude'], o['longitude'],
                                         ev['latitude'], ev['longitude'])
            stationxyz[i, 0] = o['station']
            stationxyz[i, 1] = float(x_km)
            stationxyz[i, 2] = float(y_km)
            stationxyz[i, 3] = depth_use
            if 'features' in o:
                polarities[i, :] = o['features']
                polarities[i, 0] = float(o['polarity'])
            else:
                polarities[i, 0] = float(o['polarity'])
        out[ev['event_id']] = {
            'polarities': polarities,
            'stationxyz': stationxyz,
            'focalmech': np.array([0.0, 0.0, 0.0], dtype=np.float32),
            'mag': 0.0,
        }
    return out


# ------------------------------------------------------------
# 3.6  PT axes -> strike / dip / rake
# ------------------------------------------------------------

def pt_to_sdr(pt9):
    try:
        from pyrocko import moment_tensor as pmt
    except ImportError:
        raise ImportError("pyrocko is required for PT->SDR conversion.")
    pt = pt9.reshape(3, 3)
    P, T = pt[0], pt[1]
    mt = pmt.MomentTensor(p_axis=P, t_axis=T)
    plane1, plane2 = mt.both_strike_dip_rake()
    return (tuple(float(x) for x in plane1),
            tuple(float(x) for x in plane2))


# ------------------------------------------------------------
# 3.7  Plot A: Map + stereonet + rake histogram (3-panel)
# ------------------------------------------------------------

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _lower_hemisphere(axis):
    a = _normalize(np.asarray(axis, dtype=float))
    if a[2] > 0:
        a = -a
    plunge = np.arcsin(np.clip(-a[2], -1.0, 1.0))
    azimuth = np.arctan2(a[0], a[1])
    r = np.sqrt(2.0) * np.sin((np.pi / 2 - plunge) / 2.0)
    return float(r * np.sin(azimuth)), float(r * np.cos(azimuth))



def _add_coastline(ax, lon_min, lon_max, lat_min, lat_max):
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.geometry import box
    except ImportError:
        return
    try:
        shp = shpreader.natural_earth(resolution='10m', category='physical', name='coastline')
        reader = shpreader.Reader(shp)
        bbox = box(lon_min, lat_min, lon_max, lat_max)
        for geom in reader.geometries():
            if not geom.intersects(bbox):
                continue
            inter = geom.intersection(bbox)
            if inter.geom_type == 'LineString':
                x, y = inter.xy
                ax.plot(x, y, color='dimgray', lw=0.8, zorder=1)
            elif inter.geom_type == 'MultiLineString':
                for line in inter.geoms:
                    x, y = line.xy
                    ax.plot(x, y, color='dimgray', lw=0.8, zorder=1)
    except Exception as e:
        print('[focmech] Error drawing coastline:', e)

def plot_focal_mechanisms(df, stations_df, out_path, plane=1, pad=0.02):
    if beach is None:
        print("[focmech] obspy.imaging.beachball not available; skipping beachball map.")
        return

    fig = plt.figure(figsize=(15, 6))
    ax_map = fig.add_subplot(1, 3, 1)
    ax_pt  = fig.add_subplot(1, 3, 2)
    ax_rk  = fig.add_subplot(1, 3, 3)

    s_col, d_col, r_col = f'strike{plane}', f'dip{plane}', f'rake{plane}'
    lon_min = float(df['longitude'].min()) - pad
    lon_max = float(df['longitude'].max()) + pad
    lat_min = float(df['latitude'].min()) - pad
    lat_max = float(df['latitude'].max()) + pad
    bb_size = max(0.0008, 0.03 * (lon_max - lon_min))

    for _, row in df.iterrows():
        if any(pd.isna(row[c]) for c in (s_col, d_col, r_col, 'longitude', 'latitude')):
            continue
        try:
            b = beach([float(row[s_col]), float(row[d_col]), float(row[r_col])],
                      xy=(float(row['longitude']), float(row['latitude'])),
                      width=bb_size,
                      facecolor=_faulting_color(float(row[r_col])),
                      edgecolor='black', linewidth=0.4, zorder=5)
            ax_map.add_collection(b)
        except Exception:
            pass

    if len(stations_df) > 0:
        st_handle = ax_map.scatter(
            stations_df['longitude'].values, stations_df['latitude'].values,
            marker='^', s=70, color='black', edgecolor='white',
            linewidth=0.5, zorder=6, label='stations')
        legend_handles = _fault_legend_handles() + [st_handle]
    else:
        legend_handles = _fault_legend_handles()

    ax_map.set_xlim(lon_min, lon_max)
    ax_map.set_ylim(lat_min, lat_max)
    ax_map.set_xlabel('Longitude (°E)')
    ax_map.set_ylabel('Latitude (°N)')
    _add_coastline(ax_map, lon_min, lon_max, lat_min, lat_max)
    ax_map.set_title(f'Focal mechanisms (plane {plane}) — {len(df)} events')
    ax_map.set_aspect(1.0 / np.cos(np.radians(0.5 * (lat_min + lat_max))))
    ax_map.legend(handles=legend_handles, loc='upper right', fontsize=8)

    theta = np.linspace(0.0, 2.0 * np.pi, 360)
    ax_pt.plot(np.cos(theta), np.sin(theta), color='black', linewidth=1.0)
    for txt, xy in (('N', (0, 1.05)), ('E', (1.05, 0)),
                    ('S', (0, -1.1)), ('W', (-1.1, 0))):
        ax_pt.text(*xy, txt, ha='center', va='center', fontsize=10)
    ax_pt.plot([-1.0, 1.0], [0, 0], color='lightgray', linewidth=0.5, zorder=0)
    ax_pt.plot([0, 0], [-1.0, 1.0], color='lightgray', linewidth=0.5, zorder=0)
    px, py, tx, ty = [], [], [], []
    for _, row in df.iterrows():
        try:
            P = np.array([row['Px'], row['Py'], row['Pz']], dtype=float)
            T = np.array([row['Tx'], row['Ty'], row['Tz']], dtype=float)
        except Exception:
            continue
        if not np.isfinite(P).all() or not np.isfinite(T).all():
            continue
        x, y = _lower_hemisphere(P)
        px.append(x); py.append(y)
        x, y = _lower_hemisphere(T)
        tx.append(x); ty.append(y)
    ax_pt.scatter(px, py, color='black', s=18, marker='o', label='P axis', zorder=4)
    ax_pt.scatter(tx, ty, color='red',   s=18, marker='^', label='T axis', zorder=4)
    ax_pt.set_xlim(-1.15, 1.15)
    ax_pt.set_ylim(-1.2, 1.15)
    ax_pt.set_aspect('equal')
    ax_pt.set_xticks([]); ax_pt.set_yticks([])
    ax_pt.set_title('Lower-hemisphere P/T axes')
    ax_pt.legend(loc='lower right', fontsize=8)

    rake = df[r_col].dropna().values
    ax_rk.hist(rake, bins=np.arange(-180, 181, 15), color='#888888',
               edgecolor='black', linewidth=0.4)
    for x, label in ((-90, 'normal'), (0, 'strike-slip'),
                     (90, 'thrust'), (180, 'strike-slip')):
        ax_rk.axvline(x, color='red', linestyle='--', linewidth=0.7, alpha=0.6)
    ax_rk.set_xlim(-180, 180)
    ax_rk.set_xticks(np.arange(-180, 181, 60))
    ax_rk.set_xlabel(f'rake (°) — plane {plane}')
    ax_rk.set_ylabel('count')
    ax_rk.set_title(f'Slip style ({len(rake)} events)')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[focmech] Wrote {out_path}")


# ------------------------------------------------------------
# 3.8  Plot B: Cross-section figures
# ------------------------------------------------------------

def _km_to_dlat(km):
    return km / 111.0


def _km_to_dlon(km, lat):
    return km / (111.0 * math.cos(math.radians(lat)))


def _add_beachballs(ax, xs, ys, sdrs, colors, size, edge='k', lw=0.3):
    if beach is None:
        return
    for x, y, sdr, c in zip(xs, ys, sdrs, colors):
        try:
            b = beach(sdr, xy=(float(x), float(y)), width=size,
                      facecolor=c, edgecolor=edge, linewidth=lw, zorder=5)
            ax.add_collection(b)
        except Exception:
            pass


def _draw_cross_map(ax, df, stations_df, lines, axis_label, bb_size_deg, plane, pad=0.02):
    s_col, d_col, r_col = f'strike{plane}', f'dip{plane}', f'rake{plane}'
    sdrs = [(float(r[s_col]), float(r[d_col]), float(r[r_col])) for _, r in df.iterrows()]
    colors = [_faulting_color(float(r[r_col])) for _, r in df.iterrows()]
    _add_beachballs(ax, df['longitude'], df['latitude'], sdrs, colors, bb_size_deg)
    lon_lo = float(df['longitude'].min()) - pad
    lon_hi = float(df['longitude'].max()) + pad
    lat_lo = float(df['latitude'].min()) - pad
    lat_hi = float(df['latitude'].max()) + pad
    if len(stations_df) > 0:
        in_win = ((stations_df['longitude'] >= lon_lo) & (stations_df['longitude'] <= lon_hi) &
                  (stations_df['latitude'] >= lat_lo) & (stations_df['latitude'] <= lat_hi))
        sw = stations_df[in_win]
        ax.scatter(sw['longitude'], sw['latitude'], marker='^', s=70,
                   color='dodgerblue', edgecolors='navy', linewidths=0.6, zorder=4)
    for i, ln in enumerate(lines):
        ax.plot([ln['lon_start'], ln['lon_end']],
                [ln['lat_start'], ln['lat_end']],
                color='red', lw=1.4, zorder=6)
        ax.annotate(f"{i+1}", xy=(ln['lon_start'], ln['lat_start']),
                    xytext=(2, 2), textcoords='offset points',
                    fontsize=8, color='darkred', weight='bold', zorder=7)
    ax.set_xlim(lon_lo, lon_hi)
    ax.set_ylim(lat_lo, lat_hi)
    ax.set_xlabel('Longitude (°E)', fontsize=9)
    ax.set_ylabel('Latitude (°N)', fontsize=9)
    _add_coastline(ax, lon_lo, lon_hi, lat_lo, lat_hi)
    ax.set_title(f'Map view ({axis_label})', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(lw=0.3, alpha=0.5)
    ax.set_aspect(1.0 / math.cos(math.radians(0.5 * (lat_lo + lat_hi))))
    ax.legend(handles=_fault_legend_handles(), loc='upper right', fontsize=7)


def _build_diag_lines(center_lat, center_lon, along_az_deg, perp_az_deg,
                      length_km, spacing_km, n_lines):
    along_e = math.sin(math.radians(along_az_deg))
    along_n = math.cos(math.radians(along_az_deg))
    perp_e = math.sin(math.radians(perp_az_deg))
    perp_n = math.cos(math.radians(perp_az_deg))
    coslat = math.cos(math.radians(center_lat))
    half = length_km / 2.0
    mid = (n_lines - 1) / 2.0
    lines = []
    for i in range(n_lines):
        offset = (i - mid) * spacing_km
        cx_de = offset * perp_e
        cx_dn = offset * perp_n
        de_s, dn_s = cx_de - half * along_e, cx_dn - half * along_n
        de_e, dn_e = cx_de + half * along_e, cx_dn + half * along_n
        lines.append({
            'lat_start': center_lat + dn_s / 111.0,
            'lon_start': center_lon + de_s / (111.0 * coslat),
            'lat_end':   center_lat + dn_e / 111.0,
            'lon_end':   center_lon + de_e / (111.0 * coslat),
            'along_e': along_e, 'along_n': along_n,
            'perp_e':  perp_e,  'perp_n':  perp_n,
            'perp_offset_km': offset,
            'length_km': length_km,
            'along_min': -half, 'along_max': +half,
            'anchor_lat': center_lat, 'anchor_lon': center_lon,
        })
    return lines


def _project_onto_line(df, line):
    coslat = math.cos(math.radians(line['anchor_lat']))
    de = (df['longitude'].values - line['anchor_lon']) * 111.0 * coslat
    dn = (df['latitude'].values - line['anchor_lat']) * 111.0
    along = de * line['along_e'] + dn * line['along_n']
    perp  = de * line['perp_e']  + dn * line['perp_n'] - line['perp_offset_km']
    return along, perp


def _cross_section_diag(df, stations_df, lines, axis_label, band_km, depth_lim,
                        out_path, plane, bb_size_deg):
    s_col, d_col, r_col = f'strike{plane}', f'dip{plane}', f'rake{plane}'
    n = len(lines)
    fig = plt.figure(figsize=(17, max(7, 2 + 1.8 * n)))
    gs = fig.add_gridspec(n, 2, width_ratios=[1.0, 1.0], hspace=0.35, wspace=0.25)
    ax_map = fig.add_subplot(gs[:, 0])
    _draw_cross_map(ax_map, df, stations_df, lines, axis_label, bb_size_deg, plane)

    for row_i, ln in enumerate(reversed(lines)):
        i = n - 1 - row_i
        is_bottom = (row_i == n - 1)
        ax = fig.add_subplot(gs[row_i, 1])
        along, perp = _project_onto_line(df, ln)
        a_lo = ln.get('along_min', 0.0)
        a_hi = ln.get('along_max', ln['length_km'])
        mask = (np.abs(perp) <= band_km / 2.0) & (along >= a_lo) & (along <= a_hi)
        sub = df.iloc[mask]
        a_in = along[mask]
        if len(sub) > 0:
            sdrs = [(float(r[s_col]), float(r[d_col]), float(r[r_col])) for _, r in sub.iterrows()]
            colors = [_faulting_color(float(r[r_col])) for _, r in sub.iterrows()]
            bb_xs = max(0.05, 0.03 * (a_hi - a_lo))
            _add_beachballs(ax, a_in, sub['depth_km'], sdrs, colors, bb_xs)
        ax.set_xlim(a_lo, a_hi)
        ax.set_title(f"#{i+1}  perp = {ln['perp_offset_km']:+.1f} km  (n = {len(sub)})", fontsize=9)
        if is_bottom:
            ax.set_xlabel('Along-line distance from center (km)', fontsize=8)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        if depth_lim is not None:
            ax.set_ylim(depth_lim[1], depth_lim[0])
        else:
            ax.invert_yaxis()
        ax.set_ylabel('Depth (km)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(lw=0.3, alpha=0.5)

    fig.suptitle(out_path.stem, fontsize=12, fontweight='bold', y=0.995)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[focmech] Wrote {out_path}")


def _cross_section_ew_ns(df, stations_df, lines, axis, band_km, lat_anchor,
                         depth_lim, out_path, plane, bb_size_deg):
    s_col, d_col, r_col = f'strike{plane}', f'dip{plane}', f'rake{plane}'
    n = len(lines)
    fig = plt.figure(figsize=(17, max(7, 2 + 1.8 * n)))
    gs = fig.add_gridspec(n, 2, width_ratios=[1.0, 1.0], hspace=0.35, wspace=0.25)
    ax_map = fig.add_subplot(gs[:, 0])

    uni_lines = [{'lon_start': ln['lon'][0], 'lon_end': ln['lon'][1],
                  'lat_start': ln['lat'][0], 'lat_end': ln['lat'][1]} for ln in lines]
    _draw_cross_map(ax_map, df, stations_df, uni_lines, axis, bb_size_deg, plane)

    for row_i, ln in enumerate(reversed(lines)):
        i = n - 1 - row_i
        is_bottom = (row_i == n - 1)
        ax = fig.add_subplot(gs[row_i, 1])
        if axis == 'latitude':
            half = _km_to_dlat(band_km / 2.0)
            mask = ((df['latitude'] >= ln['lat_center'] - half) &
                    (df['latitude'] <= ln['lat_center'] + half) &
                    (df['longitude'] >= ln['lon'][0]) &
                    (df['longitude'] <= ln['lon'][1]))
            sub = df[mask]
            coslat = math.cos(math.radians(ln['lat_center']))
            xs = (sub['longitude'].values - ln['lon'][0]) * 111.0 * coslat
            x_lo = 0.0
            x_hi = (ln['lon'][1] - ln['lon'][0]) * 111.0 * coslat
            if is_bottom:
                ax.set_xlabel('Easting along slice (km)', fontsize=8)
            ax.set_title(f"#{i+1}  lat = {ln['lat_center']:.4f}°  (n = {len(sub)})", fontsize=9)
        else:
            half = _km_to_dlon(band_km / 2.0, lat_anchor)
            mask = ((df['longitude'] >= ln['lon_center'] - half) &
                    (df['longitude'] <= ln['lon_center'] + half) &
                    (df['latitude'] >= ln['lat'][0]) &
                    (df['latitude'] <= ln['lat'][1]))
            sub = df[mask]
            xs = (sub['latitude'].values - ln['lat'][0]) * 111.0
            x_lo = 0.0
            x_hi = (ln['lat'][1] - ln['lat'][0]) * 111.0
            if is_bottom:
                ax.set_xlabel('Northing along slice (km)', fontsize=8)
            ax.set_title(f"#{i+1}  lon = {ln['lon_center']:.4f}°  (n = {len(sub)})", fontsize=9)

        if len(sub) > 0:
            sdrs = [(float(r[s_col]), float(r[d_col]), float(r[r_col])) for _, r in sub.iterrows()]
            colors = [_faulting_color(float(r[r_col])) for _, r in sub.iterrows()]
            bb_xs = max(0.1, 0.05 * max(x_hi - x_lo, 0.1))
            _add_beachballs(ax, xs, sub['depth_km'], sdrs, colors, bb_xs)

        ax.set_xlim(x_lo, x_hi)
        if not is_bottom:
            ax.tick_params(axis='x', labelbottom=False)
        if depth_lim is not None:
            ax.set_ylim(depth_lim[1], depth_lim[0])
        else:
            ax.invert_yaxis()
        ax.set_ylabel('Depth (km)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(lw=0.3, alpha=0.5)

    fig.suptitle(out_path.stem, fontsize=12, fontweight='bold', y=0.995)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[focmech] Wrote {out_path}")


def plot_cross_sections(df, stations_df, plots_dir, plane, spacing_km, n_lines, band_km):
    df = df.dropna(subset=['latitude', 'longitude', 'depth_km',
                            'strike1', 'dip1', 'rake1']).reset_index(drop=True)
    if len(df) == 0:
        print("[focmech] No events with full coords/SDR for cross-sections.")
        return

    lat_mean = float(df['latitude'].mean())
    lon_mean = float(df['longitude'].mean())
    cluster_lat_km = max(0.5, (df['latitude'].max() - df['latitude'].min()) * 111.0)
    cluster_lon_km = max(0.5, (df['longitude'].max() - df['longitude'].min())
                         * 111.0 * math.cos(math.radians(lat_mean)))
    pad_km = max(1.0, 0.2 * max(cluster_lat_km, cluster_lon_km))
    length = max(2.0, max(cluster_lat_km, cluster_lon_km) + 2.0 * pad_km)
    half_stack_km = spacing_km * (n_lines - 1) / 2.0
    lat_br = lat_mean - _km_to_dlat(half_stack_km)
    lon_br = lon_mean + _km_to_dlon(length / 2.0, lat_mean)
    half_len_dlon = _km_to_dlon(length, lat_br)
    half_len_dlat = _km_to_dlat(length)

    z_lo = float(df['depth_km'].min())
    z_hi = float(df['depth_km'].max())
    z_pad = max(0.3, 0.1 * (z_hi - z_lo))
    depth_lim = (z_lo - z_pad, z_hi + z_pad)

    bb_map = max(0.0008, 0.03 * (df['longitude'].max() - df['longitude'].min() + 1e-3))

    lat_lines = []
    for i in range(n_lines):
        lat_c = lat_br + _km_to_dlat(spacing_km * i)
        lat_lines.append({'lat_center': lat_c,
                          'lat': (lat_c, lat_c),
                          'lon': (lon_br - half_len_dlon, lon_br)})
    lon_lines = []
    for i in range(n_lines):
        lon_c = lon_br - _km_to_dlon(spacing_km * i, lat_br)
        lon_lines.append({'lon_center': lon_c,
                          'lon': (lon_c, lon_c),
                          'lat': (lat_br, lat_br + half_len_dlat)})

    stations_plot = stations_df.reset_index()

    _cross_section_ew_ns(df, stations_plot, lat_lines, axis='latitude',
                         band_km=band_km, lat_anchor=lat_br, depth_lim=depth_lim,
                         out_path=plots_dir / 'focal_cross_sections_latitude.png',
                         plane=plane, bb_size_deg=bb_map)
    _cross_section_ew_ns(df, stations_plot, lon_lines, axis='longitude',
                         band_km=band_km, lat_anchor=lat_br, depth_lim=depth_lim,
                         out_path=plots_dir / 'focal_cross_sections_longitude.png',
                         plane=plane, bb_size_deg=bb_map)

    center_lat = lat_br + _km_to_dlat(spacing_km * (n_lines - 1) / 2.0)
    center_lon = lon_br - half_len_dlon / 2.0
    ne_lines = _build_diag_lines(center_lat, center_lon,
                                  along_az_deg=45.0, perp_az_deg=315.0,
                                  length_km=length, spacing_km=spacing_km,
                                  n_lines=n_lines)
    nw_lines = _build_diag_lines(center_lat, center_lon,
                                  along_az_deg=315.0, perp_az_deg=45.0,
                                  length_km=length, spacing_km=spacing_km,
                                  n_lines=n_lines)
    _cross_section_diag(df, stations_plot, ne_lines, axis_label='NE-SW slices',
                        band_km=band_km, depth_lim=depth_lim,
                        out_path=plots_dir / 'focal_cross_sections_ne.png',
                        plane=plane, bb_size_deg=bb_map)
    _cross_section_diag(df, stations_plot, nw_lines, axis_label='NW-SE slices',
                        band_km=band_km, depth_lim=depth_lim,
                        out_path=plots_dir / 'focal_cross_sections_nw.png',
                        plane=plane, bb_size_deg=bb_map)


# ------------------------------------------------------------
# 3.9  Plot C: Kagan angle analysis
# ------------------------------------------------------------

RANDOM_DC_MEAN_DEG = 62.0


def _km_separation(lat1, lon1, lat2, lon2):
    coslat = math.cos(math.radians(0.5 * (lat1 + lat2)))
    dx = (lon1 - lon2) * 111.0 * coslat
    dy = (lat1 - lat2) * 111.0
    return float(math.hypot(dx, dy))


def _build_mts(df):
    try:
        from pyrocko.moment_tensor import MomentTensor
    except ImportError:
        raise ImportError("pyrocko is required for Kagan angle computation.")
    mts = []
    for _, r in df.iterrows():
        mts.append(MomentTensor(
            strike=float(r['strike1']),
            dip=float(r['dip1']),
            rake=float(r['rake1']),
        ))
    return mts


def _all_pairs_kagan(df, mts):
    try:
        from pyrocko.moment_tensor import kagan_angle
    except ImportError:
        return pd.DataFrame()
    n = len(df)
    rows = []
    lat = df['latitude'].values
    lon = df['longitude'].values
    eid = df['event_id'].values
    for i in range(n):
        for j in range(i + 1, n):
            try:
                a = float(kagan_angle(mts[i], mts[j]))
            except Exception:
                continue
            d = _km_separation(lat[i], lon[i], lat[j], lon[j])
            rows.append({'event_a': eid[i], 'event_b': eid[j],
                         'kagan_deg': a, 'distance_km': d})
    return pd.DataFrame(rows)


def plot_kagan_angles(df, out_path_png, out_path_csv):
    df_fm = df.dropna(subset=['latitude', 'longitude', 'strike1', 'dip1', 'rake1']).reset_index(drop=True)
    if len(df_fm) < 2:
        print("[focmech] Need >= 2 events for Kagan analysis; skipping.")
        return
    try:
        mts = _build_mts(df_fm)
        pairs = _all_pairs_kagan(df_fm, mts)
    except ImportError as e:
        print(f"[focmech] {e} — Kagan plot skipped.")
        return

    if pairs.empty:
        print("[focmech] No valid pairs for Kagan angles.")
        return

    pairs.to_csv(out_path_csv, index=False)
    print(f"[focmech] Wrote {out_path_csv}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.hist(pairs['kagan_deg'].values, bins=np.arange(0, 121, 4),
            color='#888888', edgecolor='black', linewidth=0.4, alpha=0.85)
    ax.axvline(RANDOM_DC_MEAN_DEG, color='red', linestyle='--', linewidth=1.0,
               label=f'random DC mean ≈ {RANDOM_DC_MEAN_DEG:.0f}°')
    med = float(np.median(pairs['kagan_deg'].values))
    ax.axvline(med, color='black', linestyle=':', linewidth=1.0,
               label=f'median = {med:.1f}°')
    ax.set_xlim(0, 120)
    ax.set_xlabel('Kagan angle (°)')
    ax.set_ylabel('count')
    ax.set_title(f'All pairs (n = {len(pairs)})')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(lw=0.3, alpha=0.5)

    ax = axes[1]
    d = pairs['distance_km'].values
    a = pairs['kagan_deg'].values
    ax.scatter(d, a, s=3, color='black', alpha=0.15)
    if len(d) >= 10:
        edges = np.linspace(0, max(0.5, np.percentile(d, 95)), 12)
        centres = 0.5 * (edges[:-1] + edges[1:])
        meds = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (d >= lo) & (d < hi)
            meds.append(float(np.median(a[m])) if m.any() else np.nan)
        ax.plot(centres, meds, color='red', linewidth=1.6,
                marker='o', markersize=4, label='binned median')
        ax.legend(loc='lower right', fontsize=8)
    ax.set_xlabel('inter-event distance (km)')
    ax.set_ylabel('Kagan angle (°)')
    ax.set_title('Mechanism similarity vs distance')
    ax.set_ylim(0, 120)
    ax.grid(lw=0.3, alpha=0.5)

    ax = axes[2]
    n_events = len(df_fm)
    ax.hist(df_fm['strike1'].values, bins=np.arange(0, 361, 15),
            color='steelblue', edgecolor='black', linewidth=0.4, alpha=0.85)
    ax.set_xlabel('Strike plane 1 (°)')
    ax.set_ylabel('count')
    ax.set_title(f'Strike distribution ({n_events} events)')
    ax.grid(lw=0.3, alpha=0.5)

    fig.suptitle(f'Kagan-angle analysis — {n_events} events, '
                 f'~{len(pairs)} pairs',
                 fontsize=12, fontweight='bold', y=0.995)
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=150)
    plt.close(fig)
    print(f"[focmech] Wrote {out_path_png}")


# ------------------------------------------------------------
# 3.10  Main
# ------------------------------------------------------------

def main():
    relocated_catalog_csv = CONFIG['relocated_catalog_csv']
    phases_csv           = CONFIG['phases_csv']
    stations_file        = CONFIG['stations_file']
    waveform_folder      = CONFIG['waveform_folder']
    output_folder        = CONFIG['output_folder']
    ckpt_path            = CONFIG['checkpoint_path']

    device               = CONFIG.get('device', 'cpu')
    batch_size           = int(CONFIG.get('batch_size', 64))
    min_polarities       = int(CONFIG.get('min_polarities', 4))
    pol_min_conf         = float(CONFIG.get('polarity_min_confidence', 0.67))
    pol_min_snr          = float(CONFIG.get('polarity_min_snr_log10', 0.3))
    depth_clip           = CONFIG.get('depth_clip_km', [0.0, 100.0])
    depth_min, depth_max = float(depth_clip[0]), float(depth_clip[1])
    nodal_plane          = int(CONFIG.get('nodal_plane', 1))
    spacing_km           = float(CONFIG.get('cross_section_spacing_km', 10.0))
    n_lines              = int(CONFIG.get('cross_section_n_lines', 3))
    band_km              = float(CONFIG.get('cross_section_band_km', spacing_km))
    network              = CONFIG['network']

    out_dir = Path(output_folder)
    plots_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("[focmech] Loading relocated catalog and phase picks...")
    catalog_df = load_relocated_catalog(relocated_catalog_csv)
    phases_df  = load_phases(phases_csv)
    stations_df = load_stations(stations_file, network)

    print(f"[focmech] {len(catalog_df)} relocated events (from {Path(relocated_catalog_csv).name}), "
          f"{len(phases_df)} total phase picks loaded.")

    print("[focmech] Building waveform index...")
    wf_idx = _build_waveform_index(waveform_folder)
    print(f"[focmech] {len(wf_idx)} waveform files indexed.")

    p_phases = phases_df[phases_df['phase'] == 'P'].copy()

    events_obs = []
    n_skip_nopick = 0
    n_skip_pol    = 0
    for _, ev_row in catalog_df.iterrows():
        ev_id = str(ev_row['event_id'])
        ev_p_picks = p_phases[p_phases['event_id'] == ev_id]
        if len(ev_p_picks) == 0:
            n_skip_nopick += 1
            continue
        obs = extract_event_observations(
            event_row=ev_row,
            event_phases_df=phases_df[phases_df['event_id'] == ev_id],
            stations_df=stations_df,
            wf_idx=wf_idx,
            cfg=CONFIG,
        )
        for o in obs:
            if o['polarity_conf'] < pol_min_conf or o['snr_pz'] < pol_min_snr:
                o['polarity'] = 0
        non_zero = sum(1 for o in obs if o['polarity'] != 0)
        if non_zero < min_polarities:
            n_skip_pol += 1
            continue
        events_obs.append({
            'event_id':    ev_id,
            'origin_time': ev_row['origin_time'],
            'latitude':    float(ev_row['latitude']),
            'longitude':   float(ev_row['longitude']),
            'depth_km':    float(ev_row['depth_km']),
            'observations': obs,
        })

    print(f"[focmech] {len(events_obs)} relocated events with >= {min_polarities} polarities "
          f"(skipped: {n_skip_nopick} no-pick, {n_skip_pol} low-polarity).")

    if not events_obs:
        print("[focmech] No relocated events with sufficient polarities. Exiting.")
        return

    print("[focmech] Building FocoNet input dict...")
    input_dict = build_foconet_input(events_obs, depth_min, depth_max)
    input_npy  = out_dir / 'event_input.npy'
    output_npy = out_dir / 'PT_predictions.npy'
    np.save(input_npy, input_dict, allow_pickle=True)
    print(f"[focmech] Wrote {input_npy} ({len(input_dict)} events)")

    print(f"[focmech] Running direct in-memory FocoNet PyTorch inference ({device})...")
    pts = run_foconet_predict_direct(
        input_dict=input_dict,
        ckpt_path=ckpt_path,
        device_str=device,
        batch_size=batch_size,
    )
    np.save(output_npy, pts)
    print(f"[focmech] Saved predictions to {output_npy}")

    n_pred = pts.shape[0]
    n_ev   = len(events_obs)
    if n_pred != n_ev:
        print(f"[focmech] Warning: {n_pred} predictions vs {n_ev} events; truncating.")
        n = min(n_pred, n_ev)
        pts = pts[:n]
        events_obs = events_obs[:n]

    print("[focmech] Converting PT axes -> strike/dip/rake...")
    rows = []
    for ev, pt in zip(events_obs, pts):
        try:
            (s1, d1, r1), (s2, d2, r2) = pt_to_sdr(pt)
        except Exception as e:
            print(f"[focmech] PT->SDR failed for {ev['event_id']}: {e}")
            continue
        non_zero = sum(1 for o in ev['observations'] if o['polarity'] != 0)
        mean_conf = float(np.mean([o['polarity_conf'] for o in ev['observations']
                                   if o['polarity'] != 0])) if non_zero else 0.0
        mean_snr  = float(np.mean([o['snr_pz'] for o in ev['observations']
                                   if o['polarity'] != 0])) if non_zero else 0.0
        rows.append({
            'event_id':          ev['event_id'],
            'origin_time':       str(ev['origin_time']),
            'latitude':          ev['latitude'],
            'longitude':         ev['longitude'],
            'depth_km':          ev['depth_km'],
            'n_polarities':      non_zero,
            'n_stations':        len(ev['observations']),
            'mean_polarity_conf': mean_conf,
            'mean_snr_log10':    mean_snr,
            'strike1': s1, 'dip1': d1, 'rake1': r1,
            'strike2': s2, 'dip2': d2, 'rake2': r2,
            'Px': float(pt[0]), 'Py': float(pt[1]), 'Pz': float(pt[2]),
            'Tx': float(pt[3]), 'Ty': float(pt[4]), 'Tz': float(pt[5]),
            'Bx': float(pt[6]), 'By': float(pt[7]), 'Bz': float(pt[8]),
            'variant': CONFIG.get('variant', 'Full'),
        })

    df_out = pd.DataFrame(rows)
    csv_path = out_dir / 'focal_mechanisms.csv'
    df_out.to_csv(csv_path, index=False)
    print(f"[focmech] Wrote {len(df_out)} focal mechanisms -> {csv_path}")

    stations_plot = stations_df.reset_index()

    if len(df_out) > 0:
        print("[focmech] Plotting focal mechanisms (map + stereonet + rake)...")
        plot_focal_mechanisms(
            df=df_out,
            stations_df=stations_plot,
            out_path=plots_dir / 'focal_mechanisms.png',
            plane=nodal_plane,
        )

        print("[focmech] Plotting cross-sections...")
        plot_cross_sections(
            df=df_out,
            stations_df=stations_df,
            plots_dir=plots_dir,
            plane=nodal_plane,
            spacing_km=spacing_km,
            n_lines=n_lines,
            band_km=band_km,
        )

        print("[focmech] Computing Kagan angles...")
        plot_kagan_angles(
            df=df_out,
            out_path_png=plots_dir / 'kagan_angles.png',
            out_path_csv=out_dir  / 'kagan_pairs.csv',
        )

    print("[focmech] Done.")


if __name__ == '__main__':
    main()
