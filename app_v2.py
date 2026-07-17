# -*- coding: utf-8 -*-
"""
日本産コウモリ音声識別アプリ Ver.2.0
スペクトログラムを見ながら時間・周波数範囲を指定して雑音を削除 → 種判別
"""
import pathlib, io, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy import signal as scipy_signal
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ─── パスワード認証 ───────────────────────────────────────
def check_password():
    if st.session_state.get("authenticated"):
        return True
    st.title("🦇 日本産コウモリ 音声識別アプリ Ver.2.0")
    pw = st.text_input("パスワードを入力してください", type="password")
    if pw:
        if pw == st.secrets.get("APP_PASSWORD", ""):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("パスワードが違います")
    st.stop()

check_password()

# ─── パス設定 ────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
CLASS_JSON = BASE_DIR / "models" / "class_names.json"
IMG_SIZE   = 224
N_FFT      = 2048
HOP        = N_FFT // 4
FREQ_MIN   = 10    # kHz
FREQ_MAX   = 130   # kHz
TOP_K      = 5

# ─── 種の補足情報 ─────────────────────────────────────────
SPECIES_INFO = {
    "アブラコウモリ":      {"latin": "Alionoctula abramus",           "en": "Japanese Pipistrelle"},
    "カグヤコウモリ":      {"latin": "Myotis longicaudatus",          "en": "Long-tailed Myotis"},
    "キクガシラコウモリ":  {"latin": "Rhinolophus nippon",            "en": "Greater Japanese Horseshoe Bat"},
    "キタクビワコウモリ":  {"latin": "Cnephaeus nilssonii",           "en": "Northern Serotine"},
    "クビワコウモリ":      {"latin": "Cnephaeus japonensis",          "en": "Japanese Serotine"},
    "クロホオヒゲコウモリ":{"latin": "Myotis pruinosus",              "en": "Frosted Myotis"},
    "コキクガシラコウモリ":{"latin": "Rhinolophus cornutus",          "en": "Little Japanese Horseshoe Bat"},
    "コテングコウモリ":    {"latin": "Murina ussuriensis",            "en": "Ussuri Tube-nosed Bat"},
    "コヤマコウモリ":      {"latin": "Nyctalus furvus",               "en": "Japanese Noctule"},
    "チチブコウモリ":      {"latin": "Barbastella pacifica",          "en": "Japanese Barbastelle"},
    "テングコウモリ":      {"latin": "Murina hilgendorfi",            "en": "Hilgendorf's Tube-nosed Bat"},
    "ドーベントンコウモリ":{"latin": "Myotis petax",                  "en": "Eastern Water Bat"},
    "ニホンウサギコウモリ":{"latin": "Plecotus sacrimontis",          "en": "Japanese Long-eared Bat"},
    "ノレンコウモリ":      {"latin": "Myotis bombinus",               "en": "Far Eastern Myotis"},
    "ヒナコウモリ":        {"latin": "Vespertilio sinensis",          "en": "Asian Particolored Bat"},
    "ヒメヒナコウモリ":    {"latin": "Vespertilio murinus",           "en": "Eurasian Particolored Bat"},
    "ヒメホオヒゲコウモリ":{"latin": "Myotis ikonnikovi",             "en": "Ikonnikov's Myotis"},
    "モモジロコウモリ":    {"latin": "Myotis macrodactylus",          "en": "Big-footed Myotis"},
    "モリアブラコウモリ":  {"latin": "Alionoctula endoi",             "en": "Endo's Pipistrelle"},
    "ヤマコウモリ":        {"latin": "Nyctalus aviator",              "en": "Bird-like Noctule"},
    "ユビナガコウモリ":    {"latin": "Miniopterus fuliginosus",       "en": "Asian Long-fingered Bat"},
    "オヒキコウモリ":      {"latin": "Tadarida insignis",             "en": "Japanese Free-tailed Bat"},
}

SPECIES_MERGE = {
    "ウサギコウモリ": "ニホンウサギコウモリ",
}

# ─── モデル読み込み（キャッシュ）────────────────────────
@st.cache_resource
def load_model():
    with open(CLASS_JSON, encoding="utf-8") as f:
        idx_to_class = json.load(f)
    num_classes = len(idx_to_class)
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(m.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    m.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
    m.eval()
    return m, idx_to_class

# ─── 音声処理関数 ─────────────────────────────────────────
def wav_to_spectrogram(audio_data, sr):
    f, t, Sxx = scipy_signal.spectrogram(
        audio_data, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann"
    )
    Sxx_dB = 10.0 * np.log10(Sxx + 1e-10)
    mask = (f / 1000 >= FREQ_MIN) & (f / 1000 <= FREQ_MAX)
    return f[mask] / 1000, t, Sxx_dB[mask]


def spectrogram_to_image(f_kHz, t, Sxx_dB, erase_regions=None, duration_sec=None):
    """スペクトログラム → PIL Image（消去領域を赤枠でオーバーレイ）"""
    dpi = 100
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=dpi)
    ax.pcolormesh(t * 1000, f_kHz, Sxx_dB, shading="auto", cmap="inferno",
                  vmin=np.percentile(Sxx_dB, 5), vmax=np.percentile(Sxx_dB, 99))

    # 消去領域を赤い半透明矩形で表示
    if erase_regions:
        import matplotlib.patches as mpatches
        for (ts, te, fs, fe) in erase_regions:
            rect = mpatches.Rectangle(
                (ts * 1000, fs), (te - ts) * 1000, fe - fs,
                linewidth=1.5, edgecolor="red", facecolor="red", alpha=0.25
            )
            ax.add_patch(rect)

    ax.set_xlabel("時間 (ms)", fontsize=8)
    ax.set_ylabel("周波数 (kHz)", fontsize=8)
    ax.tick_params(labelsize=7)
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def apply_erase_regions(audio_data, sr, regions):
    """指定された時間・周波数領域をゼロ化（FFTで該当帯域を消去）"""
    audio_out = audio_data.copy()
    n_samples = len(audio_data)
    for (t_start, t_end, f_lo, f_hi) in regions:
        s0 = max(0, int(t_start * sr))
        s1 = min(n_samples, int(t_end * sr))
        if s1 <= s0:
            continue
        segment = audio_out[s0:s1].copy()
        freqs_fft = np.fft.rfftfreq(len(segment), d=1.0 / sr) / 1000
        spectrum  = np.fft.rfft(segment)
        erase_mask = (freqs_fft >= f_lo) & (freqs_fft <= f_hi)
        spectrum[erase_mask] = 0
        audio_out[s0:s1] = np.fft.irfft(spectrum, n=len(segment))
    return audio_out


def spectrogram_to_tensor_from_data(audio_data, sr):
    f_kHz, t, Sxx_dB = wav_to_spectrogram(audio_data, sr)
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    ax.pcolormesh(np.arange(Sxx_dB.shape[1]), np.arange(Sxx_dB.shape[0]),
                  Sxx_dB, shading="auto", cmap="inferno",
                  vmin=np.percentile(Sxx_dB, 5), vmax=np.percentile(Sxx_dB, 99))
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)


def predict(model, idx_to_class, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    topk_probs, topk_idx = torch.topk(probs, TOP_K)
    results = []
    for prob, idx in zip(topk_probs.tolist(), topk_idx.tolist()):
        name = idx_to_class[str(idx)]
        name = SPECIES_MERGE.get(name, name)
        results.append({"species": name, "prob": prob})
    return results


def audio_to_wav_bytes(audio_data, sr):
    buf = io.BytesIO()
    sf.write(buf, audio_data, sr, format="WAV")
    buf.seek(0)
    return buf.read()


# ─── UI ─────────────────────────────────────────────────
st.set_page_config(
    page_title="日本産コウモリ音声識別 Ver.2.0",
    page_icon="🦇",
    layout="centered",
)

st.title("🦇 日本産コウモリ 音声識別アプリ Ver.2.0")
st.caption("スペクトログラムで雑音領域を範囲指定して削除してから種判別できます")

with st.spinner("モデルを読み込んでいます..."):
    model, idx_to_class = load_model()
st.success("モデル準備完了（22 種対応）")
st.divider()

# ─── STEP 1：ファイルアップロード ────────────────────────
st.subheader("Step 1　WAVファイルをアップロード")
uploaded = st.file_uploader("WAVファイルを選択してください", type=["wav", "WAV"])

if uploaded is None:
    st.stop()

original_name  = pathlib.Path(uploaded.name).stem
audio_bytes    = uploaded.read()
audio_data, sr = sf.read(io.BytesIO(audio_bytes))
if audio_data.ndim > 1:
    audio_data = audio_data[:, 0]
audio_data   = audio_data.astype(np.float32)
duration_sec = len(audio_data) / sr
duration_ms  = duration_sec * 1000

st.audio(audio_bytes, format="audio/wav")
col1, col2 = st.columns(2)
col1.metric("サンプルレート", f"{sr / 1000:.0f} kHz")
col2.metric("録音時間", f"{duration_ms:.1f} ms")

# ─── STEP 2：スペクトログラム表示＋消去領域指定 ──────────
st.divider()
st.subheader("Step 2　消去する領域を指定する")

if "erase_regions" not in st.session_state:
    st.session_state.erase_regions = []

# スペクトログラム生成・表示（消去領域を赤枠でオーバーレイ）
f_kHz, t_arr, Sxx_dB = wav_to_spectrogram(audio_data, sr)
spec_img = spectrogram_to_image(f_kHz, t_arr, Sxx_dB,
                                erase_regions=st.session_state.erase_regions,
                                duration_sec=duration_sec)
st.image(spec_img, use_container_width=True,
         caption="スペクトログラム（10〜130 kHz）　赤枠＝消去予定領域")

st.markdown("**消去したい雑音の時間・周波数範囲を指定して「追加」してください。複数指定可能です。**")

with st.form("add_region_form", clear_on_submit=True):
    col_t, col_f = st.columns(2)
    with col_t:
        t_lo = st.number_input("開始時間 (ms)", min_value=0.0,
                               max_value=float(duration_ms), value=0.0, step=1.0)
        t_hi = st.number_input("終了時間 (ms)", min_value=0.0,
                               max_value=float(duration_ms), value=float(duration_ms), step=1.0)
    with col_f:
        f_lo = st.number_input("最低周波数 (kHz)", min_value=float(FREQ_MIN),
                               max_value=float(FREQ_MAX), value=float(FREQ_MIN), step=1.0)
        f_hi = st.number_input("最高周波数 (kHz)", min_value=float(FREQ_MIN),
                               max_value=float(FREQ_MAX), value=float(FREQ_MAX), step=1.0)
    add_btn = st.form_submit_button("➕ この領域を消去リストに追加", type="primary")
    if add_btn:
        if t_hi <= t_lo:
            st.error("終了時間は開始時間より大きくしてください。")
        elif f_hi <= f_lo:
            st.error("最高周波数は最低周波数より大きくしてください。")
        else:
            st.session_state.erase_regions.append(
                (t_lo / 1000, t_hi / 1000, f_lo, f_hi)
            )
            st.rerun()

# 消去リスト表示
if st.session_state.erase_regions:
    st.markdown("**消去予定の領域一覧（赤枠）:**")
    for i, (ts, te, fs, fe) in enumerate(st.session_state.erase_regions, 1):
        st.markdown(
            f"　{i}. 時間 **{ts*1000:.0f} 〜 {te*1000:.0f} ms** "
            f"/ 周波数 **{fs:.0f} 〜 {fe:.0f} kHz**"
        )
    if st.button("🗑　消去リストをすべてクリア"):
        st.session_state.erase_regions = []
        st.rerun()
else:
    st.info("消去領域が未設定です。雑音がない場合はそのまま Step 3 へ進んでください。")

# ─── STEP 3：修正後の種判別 ──────────────────────────────
st.divider()
st.subheader("Step 3　種判別を実行する")

has_regions = len(st.session_state.erase_regions) > 0
if has_regions:
    st.info(f"{len(st.session_state.erase_regions)} 件の消去領域が設定されています。修正後の音声で種判別します。")
else:
    st.info("消去領域なし。元の音声でそのまま種判別します。")

if st.button("▶ 種判別を実行", type="primary"):
    with st.spinner("処理中..."):
        if has_regions:
            audio_processed = apply_erase_regions(
                audio_data, sr, st.session_state.erase_regions
            )
        else:
            audio_processed = audio_data.copy()

        if has_regions:
            st.subheader("修正後のスペクトログラム")
            f2, t2, S2 = wav_to_spectrogram(audio_processed, sr)
            img2 = spectrogram_to_image(f2, t2, S2)
            st.image(img2, use_container_width=True)

        tensor  = spectrogram_to_tensor_from_data(audio_processed, sr)
        results = predict(model, idx_to_class, tensor)

    top  = results[0]
    sp   = top["species"]
    conf = top["prob"]

    st.markdown(f"## 推定種：**{sp}**")
    info = SPECIES_INFO.get(sp, {})
    if info:
        st.markdown(f"*{info.get('latin', '')}* / {info.get('en', '')}")
    st.progress(conf, text=f"確信度：{conf:.1%}")

    st.subheader("上位 5 候補")
    for r in results:
        st.progress(min(r["prob"], 1.0), text=f"{r['species']}  {r['prob']:.1%}")

    if has_regions:
        st.divider()
        st.subheader("修正済み音声ファイルのダウンロード")
        modified_wav  = audio_to_wav_bytes(audio_processed, sr)
        download_name = f"{original_name}修正.wav"
        st.download_button(
            label=f"💾 {download_name} をダウンロード",
            data=modified_wav,
            file_name=download_name,
            mime="audio/wav",
        )

    st.divider()
    st.info(
        "**ご注意** : このモデルは試験的なものです。"
        "確信度が低い場合（目安: 50% 未満）は、専門家による確認をお勧めします。"
        f"  \n学習データ: 日本産 22 種・2,097 録音（Ver.1.7）"
    )
