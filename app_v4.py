# -*- coding: utf-8 -*-
"""
日本産コウモリ音声識別アプリ Ver.4.0（実験版）
Google Perch（Perch2ONNX、bioacoustics-model-zoo経由）の音響埋め込みを転移学習に
用いた方式。本番app.py（Ver.3.2 階層分類CNN）とは独立した別アプリとして、
比較検証のために公開する。

venv_perch環境で実行すること:
  streamlit run app_v4.py
"""
import pathlib, io, warnings, tempfile
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import soundfile as sf
import joblib
from scipy import signal as scipy_signal
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ─── パスワード認証（本番appと同じ仕組み） ─────────────────
def check_password():
    if st.session_state.get("authenticated"):
        return True
    st.title("🦇 日本産コウモリ 音声識別アプリ Ver.4.0（実験版）")
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
BASE_DIR    = pathlib.Path(__file__).parent
MODEL_DIR   = BASE_DIR / "models"
COVER_IMAGE = BASE_DIR / "cover_bat.jpg"
N_FFT       = 2048
TARGET_SR   = 44100   # ピッチシフト後のサンプルレート（Ghani et al. 2023方式）
TOP_K       = 5

# 場所ベース評価の正直な数値（README参照）
LOCATION_BASED_ACC_MULTI = 0.496   # 複数地点17種
SAMELOC_ACC_SINGLE = 0.952         # 単一地点5種（同一地点内のみ、参考値）

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

SINGLE_LOCATION_SPECIES = {"オヒキコウモリ", "キタクビワコウモリ", "コヤマコウモリ",
                            "ドーベントンコウモリ", "ヒメヒナコウモリ"}

# ─── モデル読み込み（キャッシュ） ───────────────────────────
@st.cache_resource
def load_models():
    import bioacoustics_model_zoo as bmz
    perch = bmz.Perch2ONNX(headless=True)
    clf = joblib.load(MODEL_DIR / "perch_classifier.joblib")
    return perch, clf


# ─── 前処理・推論 ────────────────────────────────────────
def pitch_shift_bytes_to_tempfile(audio_bytes):
    """アップロードされたWAVバイト列を、サンプルレートのラベル付け替えのみで
    44.1kHz相当（可聴域）に変換し、一時ファイルとして保存する。
    （タイム・エクスパンションと同義。波形データ自体は無加工）"""
    data, sr_orig = sf.read(io.BytesIO(audio_bytes))
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, data, TARGET_SR)
    duration_shifted = len(data) / TARGET_SR
    return tmp.name, sr_orig, len(data) / sr_orig, duration_shifted


def spectrogram_of_bytes(audio_bytes):
    data, sr = sf.read(io.BytesIO(audio_bytes))
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    hop = N_FFT // 4
    f, t, Sxx = scipy_signal.spectrogram(
        data, fs=sr, nperseg=N_FFT, noverlap=N_FFT - hop, window="hann", scaling="spectrum"
    )
    Sxx_dB = 10.0 * np.log10(Sxx + 1e-10)
    mask = (f / 1000 >= 10) & (f / 1000 <= 130)
    return f[mask] / 1000, t, Sxx_dB[mask]


def spectrogram_to_pil(f_kHz, t, Sxx_dB):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.pcolormesh(t, f_kHz, Sxx_dB, shading="auto", cmap="inferno",
                  vmin=np.percentile(Sxx_dB, 5), vmax=np.percentile(Sxx_dB, 99))
    ax.set_xlabel("時間 (秒)")
    ax.set_ylabel("周波数 (kHz)")
    ax.set_title("スペクトログラム（元の超音波録音）")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def predict(perch, clf, wav_path):
    emb_df = perch.embed([wav_path])
    embed_cols = [c for c in emb_df.columns]
    feature = emb_df[embed_cols].values.mean(axis=0).reshape(1, -1)

    probs = clf.predict_proba(feature)[0]
    classes = clf.classes_
    order = np.argsort(probs)[::-1][:TOP_K]
    results = [{"species": classes[i], "prob": probs[i]} for i in order]
    return results, emb_df.shape[0]


# ─── UI ─────────────────────────────────────────────────
st.set_page_config(
    page_title="日本産コウモリ音声識別 Ver.4.0",
    page_icon="🦇",
    layout="centered",
)

if COVER_IMAGE.exists():
    st.image(str(COVER_IMAGE), use_container_width=True)

st.title("🦇 日本産コウモリ 音声識別アプリ Ver.4.0（実験版）")
st.caption("Google Perch（Perch2ONNX）の音響埋め込みを転移学習した実験版。"
           "本番のVer.3.2（階層分類CNN）とは独立した別アプリです。")

st.warning(
    "**このアプリはVer.3.2との比較検証用の実験版です。**  \n"
    "場所ベースの正しい評価（未知の調査地での実力）では、複数地点データがある17種で"
    f"**精度{LOCATION_BASED_ACC_MULTI:.1%}**でした。データが1地点のみの5種"
    "（オヒキコウモリ・キタクビワコウモリ・コヤマコウモリ・ドーベントンコウモリ・"
    f"ヒメヒナコウモリ）は同一地点内評価で精度{SAMELOC_ACC_SINGLE:.1%}ですが、"
    "未知の場所での性能は未検証です。"
)

with st.spinner("モデルを読み込んでいます（初回は数十秒かかります）..."):
    perch, clf = load_models()

st.success(f"モデル準備完了（22種対応・Ver.4.0 Perch転移学習・学習データ2,514録音）")
st.divider()

uploaded = st.file_uploader("WAVファイルをアップロード", type=["wav", "WAV"])

if uploaded is not None:
    audio_bytes = uploaded.read()
    st.audio(audio_bytes, format="audio/wav")

    with st.spinner("解析中..."):
        try:
            f_kHz, t, Sxx_dB = spectrogram_of_bytes(audio_bytes)
            spec_img = spectrogram_to_pil(f_kHz, t, Sxx_dB)
            tmp_path, sr_orig, dur_orig, dur_shifted = pitch_shift_bytes_to_tempfile(audio_bytes)
            results, n_windows = predict(perch, clf, tmp_path)
        except Exception as e:
            st.error(f"解析エラー: {e}")
            st.stop()

    col1, col2 = st.columns(2)
    col1.metric("元サンプルレート", f"{sr_orig / 1000:.0f} kHz")
    col2.metric("録音時間", f"{dur_orig:.2f} 秒")

    st.divider()

    top = results[0]
    sp = top["species"]
    conf = top["prob"]

    st.markdown(f"## 推定種：**{sp}**")
    info = SPECIES_INFO.get(sp, {})
    if info:
        st.markdown(f"*{info.get('latin', '')}* / {info.get('en', '')}")

    st.progress(min(float(conf), 1.0), text=f"確信度：{conf:.1%}")

    if sp in SINGLE_LOCATION_SPECIES:
        st.warning(
            f"⚠️ {sp}は学習データが単一の調査地のみのため、"
            "この判定の信頼性は他地点での検証ができていません。特に慎重な確認を推奨します。"
        )
    if conf < 0.4:
        st.warning("確信度が低いため、種の識別は不確実です。専門家による確認を強く推奨します。")

    st.subheader("上位候補")
    for r in results:
        st.progress(min(float(r["prob"]), 1.0), text=f"{r['species']}  {r['prob']:.1%}")

    st.divider()
    st.subheader("スペクトログラム（10〜130 kHz、元の超音波録音）")
    st.image(spec_img, use_container_width=True)

    st.info(
        "**ご注意**：このアプリはGoogle Perch（鳥類音響埋め込みモデルPerch2.0のONNX版）を"
        "コウモリ音に転移学習した実験版です。"
        "  \nコウモリの超音波コールは、サンプルレート変換によるピッチシフト"
        "（タイム・エクスパンションと同義）で可聴域相当に変換してからPerchに入力しています。"
        "  \n場所を考慮しない同一地点内評価では高精度ですが、真に未知の調査地での精度は"
        f"約{LOCATION_BASED_ACC_MULTI:.0%}（複数地点データがある17種の場所ベース評価）に"
        "留まります。研究・比較検証用としてご利用ください。"
        "  \n学習データ: 日本産 22種・2,514録音（Ver.4.0、usableデータ全件）"
    )
