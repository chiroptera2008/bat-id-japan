# -*- coding: utf-8 -*-
"""
日本産コウモリ音声識別アプリ Ver.2.0
スペクトログラム上で消しゴム編集 → 修正WAV保存 → 種判別
"""
import pathlib, io, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torchvision import models, transforms
from scipy import signal as scipy_signal
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

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
CANVAS_W   = 700   # スペクトログラム表示幅（px）
CANVAS_H   = 350   # スペクトログラム表示高さ（px）

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
    """音声データ → スペクトログラム（周波数配列・時間配列・dB行列）"""
    f, t, Sxx = scipy_signal.spectrogram(
        audio_data, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann"
    )
    Sxx_dB = 10.0 * np.log10(Sxx + 1e-10)
    mask = (f / 1000 >= FREQ_MIN) & (f / 1000 <= FREQ_MAX)
    return f[mask] / 1000, t, Sxx_dB[mask]


def spectrogram_to_image(f_kHz, t, Sxx_dB, width=CANVAS_W, height=CANVAS_H):
    """スペクトログラム → PIL Image（指定サイズ）"""
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.pcolormesh(t * 1000, f_kHz, Sxx_dB, shading="auto", cmap="inferno",
                  vmin=np.percentile(Sxx_dB, 5), vmax=np.percentile(Sxx_dB, 99))
    ax.set_xlabel("時間 (ms)", fontsize=8)
    ax.set_ylabel("周波数 (kHz)", fontsize=8)
    ax.tick_params(labelsize=7)
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    img = img.resize((width, height), Image.LANCZOS)
    return img


def apply_eraser_mask(audio_data, sr, mask_image, t_total_sec, f_min_kHz, f_max_kHz):
    """
    消しゴムで塗られたピクセル領域を音声データに反映する。
    mask_image: RGBA PIL Image（消した部分が透明でない白ピクセル）
    """
    mask_arr = np.array(mask_image)
    h, w = mask_arr.shape[:2]

    # アルファチャンネルで「塗られた部分」を検出
    painted = mask_arr[:, :, 3] > 10

    if not painted.any():
        return audio_data.copy()

    audio_out = audio_data.copy()
    n_samples = len(audio_data)

    for px_x in range(w):
        if not painted[:, px_x].any():
            continue

        # X座標 → 時間（秒）
        t_sec = px_x / w * t_total_sec
        sample_idx = int(t_sec * sr)

        # Y方向の塗られた範囲 → 周波数範囲
        painted_rows = np.where(painted[:, px_x])[0]
        if len(painted_rows) == 0:
            continue
        y_top    = painted_rows.min()
        y_bottom = painted_rows.max()

        # Y座標 → 周波数（kHz）　上が高周波・下が低周波
        freq_top    = f_max_kHz - (y_top    / h) * (f_max_kHz - f_min_kHz)
        freq_bottom = f_max_kHz - (y_bottom / h) * (f_max_kHz - f_min_kHz)
        freq_lo = min(freq_top, freq_bottom)
        freq_hi = max(freq_top, freq_bottom)

        # その時刻のサンプル周辺で周波数フィルタリングして消去
        half = N_FFT // 2
        s0 = max(0, sample_idx - half)
        s1 = min(n_samples, sample_idx + half)
        segment = audio_out[s0:s1].copy()

        freqs_fft = np.fft.rfftfreq(len(segment), d=1.0 / sr) / 1000
        spectrum  = np.fft.rfft(segment)
        erase_mask = (freqs_fft >= freq_lo) & (freqs_fft <= freq_hi)
        spectrum[erase_mask] = 0
        audio_out[s0:s1] = np.fft.irfft(spectrum, n=len(segment))

    return audio_out


def spectrogram_to_tensor_from_data(audio_data, sr):
    """音声データ → モデル入力テンソル"""
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
    """numpy配列 → WAVバイト列"""
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
st.caption("スペクトログラム上で雑音を消しゴム削除してから種判別できます")

with st.spinner("モデルを読み込んでいます..."):
    model, idx_to_class = load_model()
st.success("モデル準備完了（22 種対応）")
st.divider()

# ─── STEP 1：ファイルアップロード ────────────────────────
st.subheader("Step 1　WAVファイルをアップロード")
uploaded = st.file_uploader("WAVファイルを選択してください", type=["wav", "WAV"])

if uploaded is None:
    st.stop()

original_name = pathlib.Path(uploaded.name).stem
audio_bytes   = uploaded.read()
audio_data, sr = sf.read(io.BytesIO(audio_bytes))
if audio_data.ndim > 1:
    audio_data = audio_data[:, 0]
audio_data = audio_data.astype(np.float32)
duration_sec = len(audio_data) / sr

st.audio(audio_bytes, format="audio/wav")
col1, col2 = st.columns(2)
col1.metric("サンプルレート", f"{sr / 1000:.0f} kHz")
col2.metric("録音時間", f"{duration_sec:.3f} 秒")

# ─── STEP 2：スペクトログラム表示＋消しゴム編集 ─────────
st.divider()
st.subheader("Step 2　スペクトログラムで雑音を消す")
st.markdown(
    "🧹 **消しゴムモード**でスペクトログラム上の雑音部分をこすってください。"
    "消した部分の音声が無音になります。"
    "雑音がない場合はそのまま Step 3 へ進んでください。"
)

# スペクトログラム画像を生成
f_kHz, t_arr, Sxx_dB = wav_to_spectrogram(audio_data, sr)
spec_img = spectrogram_to_image(f_kHz, t_arr, Sxx_dB, width=CANVAS_W, height=CANVAS_H)

# 消しゴムサイズ選択
eraser_size = st.select_slider(
    "消しゴムのサイズ",
    options=[10, 20, 30, 50, 80],
    value=30,
    format_func=lambda x: {10: "極小", 20: "小", 30: "中", 50: "大", 80: "極大"}[x],
)

# 描画キャンバス（消しゴムのみ）
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.0)",
    stroke_width=eraser_size,
    stroke_color="#FFFFFF",
    background_image=spec_img,
    update_streamlit=True,
    height=CANVAS_H,
    width=CANVAS_W,
    drawing_mode="freedraw",
    key="eraser_canvas",
)

# ─── STEP 3：修正後の種判別 ──────────────────────────────
st.divider()
st.subheader("Step 3　種判別を実行する")

has_drawing = (
    canvas_result.image_data is not None
    and canvas_result.image_data[:, :, 3].max() > 10
)

if has_drawing:
    st.info("消しゴムで編集した内容が検出されました。修正後の音声で種判別します。")
else:
    st.info("編集なし。元の音声でそのまま種判別します。")

if st.button("▶ 種判別を実行", type="primary"):
    with st.spinner("処理中..."):
        # 消しゴムが使われていたら音声を修正
        if has_drawing:
            mask_img = Image.fromarray(canvas_result.image_data.astype(np.uint8), "RGBA")
            audio_processed = apply_eraser_mask(
                audio_data, sr, mask_img,
                t_total_sec=duration_sec,
                f_min_kHz=FREQ_MIN,
                f_max_kHz=FREQ_MAX,
            )
        else:
            audio_processed = audio_data.copy()

        # 修正後スペクトログラムを表示
        if has_drawing:
            st.subheader("修正後のスペクトログラム")
            f2, t2, S2 = wav_to_spectrogram(audio_processed, sr)
            img2 = spectrogram_to_image(f2, t2, S2, width=CANVAS_W, height=CANVAS_H)
            st.image(img2, use_container_width=True)

        # 種判別
        tensor  = spectrogram_to_tensor_from_data(audio_processed, sr)
        results = predict(model, idx_to_class, tensor)

    # 結果表示
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

    st.divider()
    st.subheader("スペクトログラム（元音声）")
    st.image(spec_img, use_container_width=True)

    # 修正WAVのダウンロード
    if has_drawing:
        st.divider()
        st.subheader("修正済み音声ファイルのダウンロード")
        modified_wav = audio_to_wav_bytes(audio_processed, sr)
        download_name = f"{original_name}修正.wav"
        st.download_button(
            label=f"💾 {download_name} をダウンロード",
            data=modified_wav,
            file_name=download_name,
            mime="audio/wav",
        )

    st.info(
        "**ご注意** : このモデルは試験的なものです。"
        "確信度が低い場合（目安: 50% 未満）は、専門家による確認をお勧めします。"
        f"  \n学習データ: 日本産 22 種・2,097 録音（Ver.1.7）"
    )
