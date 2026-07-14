# -*- coding: utf-8 -*-
"""
日本産コウモリ音声識別アプリ
WAVファイルをアップロードすると、種名と確信度を表示する
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
    st.title("🦇 日本産コウモリ 音声識別アプリ")
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

# ─── 前処理関数 ──────────────────────────────────────────
def wav_to_spectrogram_array(audio_bytes):
    """WAVバイト列 → スペクトログラム配列（dB）"""
    data, sr = sf.read(io.BytesIO(audio_bytes))
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)

    f, t, Sxx = scipy_signal.spectrogram(
        data, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann"
    )
    Sxx_dB = 10.0 * np.log10(Sxx + 1e-10)
    mask = (f / 1000 >= FREQ_MIN) & (f / 1000 <= FREQ_MAX)
    return f[mask] / 1000, t, Sxx_dB[mask], sr, len(data) / sr


def spectrogram_to_pil(f_kHz, t, Sxx_dB):
    """スペクトログラム配列 → PIL Image（表示用）"""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.pcolormesh(t * 1000, f_kHz, Sxx_dB, shading="auto", cmap="inferno",
                  vmin=np.percentile(Sxx_dB, 5), vmax=np.percentile(Sxx_dB, 99))
    ax.set_xlabel("時間 (ms)")
    ax.set_ylabel("周波数 (kHz)")
    ax.set_title("スペクトログラム")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def spectrogram_to_tensor(Sxx_dB):
    """スペクトログラム配列 → モデル入力テンソル"""
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


SPECIES_MERGE = {
    "ウサギコウモリ": "ニホンウサギコウモリ",
}

def predict(model, idx_to_class, tensor):
    """推論 → Top-K 結果のリストを返す"""
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


# ─── UI ─────────────────────────────────────────────────
st.set_page_config(
    page_title="日本産コウモリ音声識別",
    page_icon="🦇",
    layout="centered",
)

st.title("🦇 日本産コウモリ 音声識別アプリ")
st.caption("D1000X（Pettersson Elektronik AB）で録音した WAV ファイルを解析します")

# モデル読み込み
with st.spinner("モデルを読み込んでいます..."):
    model, idx_to_class = load_model()

st.success(f"モデル準備完了（22 種対応）")
st.divider()

# ファイルアップロード
uploaded = st.file_uploader("WAVファイルをアップロード", type=["wav", "WAV"])

if uploaded is not None:
    audio_bytes = uploaded.read()
    st.audio(audio_bytes, format="audio/wav")

    with st.spinner("解析中..."):
        try:
            f_kHz, t, Sxx_dB, sr, duration = wav_to_spectrogram_array(audio_bytes)
            spec_img  = spectrogram_to_pil(f_kHz, t, Sxx_dB)
            tensor    = spectrogram_to_tensor(Sxx_dB)
            results   = predict(model, idx_to_class, tensor)
        except Exception as e:
            st.error(f"解析エラー: {e}")
            st.stop()

    # 録音情報
    col1, col2 = st.columns(2)
    col1.metric("サンプルレート", f"{sr / 1000:.0f} kHz")
    col2.metric("録音時間", f"{duration:.2f} 秒")

    st.divider()

    # 最有力候補
    top = results[0]
    sp  = top["species"]
    conf = top["prob"]

    conf_color = "green" if conf >= 0.7 else "orange" if conf >= 0.4 else "red"
    st.markdown(f"## 推定種：**{sp}**")
    info = SPECIES_INFO.get(sp, {})
    if info:
        st.markdown(f"*{info.get('latin', '')}* / {info.get('en', '')}")

    st.progress(conf, text=f"確信度：{conf:.1%}")

    # Top-K 結果バー
    st.subheader("上位 5 候補")
    for r in results:
        bar_val = min(r["prob"], 1.0)
        label   = f"{r['species']}  {r['prob']:.1%}"
        st.progress(bar_val, text=label)

    st.divider()

    # スペクトログラム表示
    st.subheader("スペクトログラム（10〜130 kHz）")
    st.image(spec_img, use_container_width=True)

    # 注意書き
    st.info(
        "**ご注意** : このモデルは試験的なものです。"
        "確信度が低い場合（目安: 50% 未満）は、専門家による確認をお勧めします。"
        f"  \n学習データ: 日本産 22 種・2,097 録音（Ver.1.7）"
    )
