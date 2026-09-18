# -*- coding: utf-8 -*-
"""
日本産コウモリ音声識別アプリ Ver.4.2（3方式比較版）
Ver.3.2（階層分類CNN）・Ver.4.0（Google Perch転移学習）・Ver.4.1（BirdNET転移学習）の
3方式を、同一のアップロード音声に対して同時に実行し、横並びで比較表示する。

venv_perch環境で実行すること:
  streamlit run app_compare.py
"""
import pathlib, io, json, warnings, tempfile
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import joblib
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
    st.title("🦇 日本産コウモリ音声識別 3方式比較アプリ Ver.4.2")
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
BASE_DIR     = pathlib.Path(__file__).parent
MODEL_DIR    = BASE_DIR / "models"
COVER_IMAGE  = BASE_DIR / "cover_bat.jpg"


def _find_dir_containing(filename, candidates):
    """指定ファイルを含むディレクトリを候補の中から探す。
    ローカル開発（Ver4.2_compare/models/以下に整理）と、
    GitHubリポジトリ直下（models/に集約、本番app.pyと共有）の
    どちらの配置でも動くようにするため。"""
    for d in candidates:
        if (d / filename).exists():
            return d
    return candidates[0]


# Ver.3.2のモデル（約60MB）は複製せず、リポジトリ直下のmodels/を本番app.pyと共有する
VER32_DIR = _find_dir_containing("group_model.pth", [
    MODEL_DIR / "ver32",
    MODEL_DIR,
    BASE_DIR.parent / "models",
])
VER40_DIR = _find_dir_containing("perch_classifier.joblib", [
    MODEL_DIR / "ver40_perch",
    BASE_DIR / "ver40_perch",
    MODEL_DIR,
    BASE_DIR.parent / "Ver4.0_perch" / "models",
])
VER41_DIR = _find_dir_containing("birdnet_classifier.joblib", [
    MODEL_DIR / "ver41_birdnet",
    BASE_DIR / "ver41_birdnet",
    MODEL_DIR,
    BASE_DIR.parent / "Ver4.1_birdnet" / "models",
])

IMG_SIZE   = 224
N_FFT      = 2048
HOP        = N_FFT // 4
FREQ_MIN   = 10
FREQ_MAX   = 130
TOP_K      = 5
TARGET_SR  = 44100   # ピッチシフト後（Perch/BirdNET共通、タイム・エクスパンション方式）

# 場所ベース評価の実測値（各Verのfull_location_based_evaluation.py結果）
ACC = {
    "ver32":   {"multi": None,  "single": None},   # Ver.3.2は場所ベース未評価（既知の限界）
    "ver40":   {"multi": 0.496, "single": 0.952},
    "ver41":   {"multi": 0.458, "single": 0.968},
}

# ─── 音響グループ定義（Ver.3.0/3.2 階層分類） ───────────────
GROUPS = {
    "R": ["キクガシラコウモリ", "コキクガシラコウモリ"],
    "P": ["アブラコウモリ", "モリアブラコウモリ", "ユビナガコウモリ"],
    "V": ["クビワコウモリ", "キタクビワコウモリ", "ヤマコウモリ", "コヤマコウモリ",
          "ヒナコウモリ", "ヒメヒナコウモリ", "チチブコウモリ"],
    "L": ["ニホンウサギコウモリ", "ウサギコウモリ", "テングコウモリ", "コテングコウモリ"],
    "M": ["カグヤコウモリ", "ノレンコウモリ", "モモジロコウモリ",
          "クロホオヒゲコウモリ", "ヒメホオヒゲコウモリ"],
    "T": ["オヒキコウモリ"],
}
GROUP_LABELS = {
    "R": "キクガシラ型（CF）", "P": "アブラコウモリ型（FM）",
    "V": "クビワ・ヤマ・ヒナ型（FM/CF）", "L": "ウサギ・テング型（低強度FM）",
    "M": "ホオヒゲ・ノレン型（Myotis型）", "T": "オヒキ型（自由尾）",
}
SPECIES_MERGE = {"ウサギコウモリ": "ニホンウサギコウモリ"}

SINGLE_LOCATION_SPECIES = {"オヒキコウモリ", "キタクビワコウモリ", "コヤマコウモリ",
                            "ドーベントンコウモリ", "ヒメヒナコウモリ"}

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


# ─── モデル読み込み（キャッシュ） ───────────────────────────
def _build_cnn(num_classes):
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4), nn.Linear(m.last_channel, 256), nn.ReLU(),
        nn.Dropout(p=0.3), nn.Linear(256, num_classes),
    )
    return m


@st.cache_resource
def load_ver32():
    with open(VER32_DIR / "group_classes.json", encoding="utf-8") as f:
        group_classes = json.load(f)
    group_model = _build_cnn(len(group_classes))
    group_model.load_state_dict(torch.load(str(VER32_DIR / "group_model.pth"), map_location="cpu"))
    group_model.eval()

    species_models, species_classes = {}, {}
    for g in GROUPS:
        p = VER32_DIR / f"species_model_{g}.pth"
        if p.exists():
            with open(VER32_DIR / f"species_classes_{g}.json", encoding="utf-8") as f:
                species_classes[g] = json.load(f)
            m = _build_cnn(len(species_classes[g]))
            m.load_state_dict(torch.load(str(p), map_location="cpu"))
            m.eval()
            species_models[g] = m
    return group_model, group_classes, species_models, species_classes


@st.cache_resource
def load_ver40():
    import bioacoustics_model_zoo as bmz
    perch = bmz.Perch2ONNX(headless=True)
    clf = joblib.load(VER40_DIR / "perch_classifier.joblib")
    return perch, clf


@st.cache_resource
def load_ver41():
    import bioacoustics_model_zoo as bmz
    birdnet = bmz.BirdNET()
    clf = joblib.load(VER41_DIR / "birdnet_classifier.joblib")
    return birdnet, clf


# ─── 共通前処理 ──────────────────────────────────────────
def load_wav_bytes(audio_bytes):
    data, sr = sf.read(io.BytesIO(audio_bytes))
    if data.ndim > 1:
        data = data[:, 0]
    return data.astype(np.float32), sr


def make_spectrogram(data, sr):
    f, t, Sxx = scipy_signal.spectrogram(
        data, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, window="hann", scaling="spectrum"
    )
    Sxx_dB = 10.0 * np.log10(Sxx + 1e-10)
    mask = (f / 1000 >= FREQ_MIN) & (f / 1000 <= FREQ_MAX)
    return f[mask] / 1000, t, Sxx_dB[mask]


def spectrogram_to_pil(f_kHz, t, Sxx_dB):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.pcolormesh(t, f_kHz, Sxx_dB, shading="auto", cmap="inferno",
                  vmin=np.percentile(Sxx_dB, 5), vmax=np.percentile(Sxx_dB, 99))
    # Streamlit Cloudのサーバーには日本語フォントが無く文字化けするため、軸ラベルは英語表記にする
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (kHz)")
    ax.set_title("Spectrogram (original ultrasonic recording)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def pitch_shift_to_tempfile(data, sr_orig):
    """サンプルレートのラベル付け替えのみで44.1kHz相当に変換（タイム・エクスパンション方式）"""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, data, TARGET_SR)
    return tmp.name


# ─── Ver.3.2 推論（階層分類CNN） ────────────────────────────
def spectrogram_to_tensor(Sxx_dB):
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
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)


def predict_ver32(models_tuple, tensor):
    group_model, group_classes, species_models, species_classes = models_tuple
    with torch.no_grad():
        g_probs = torch.softmax(group_model(tensor), dim=1)[0]
    g_idx = g_probs.argmax().item()
    pred_group = group_classes[str(g_idx)]
    group_conf = g_probs[g_idx].item()

    if pred_group in species_models:
        with torch.no_grad():
            s_probs = torch.softmax(species_models[pred_group](tensor), dim=1)[0]
        k = min(TOP_K, s_probs.shape[0])
        topk_probs, topk_idx = torch.topk(s_probs, k)
        results = []
        for prob, idx in zip(topk_probs.tolist(), topk_idx.tolist()):
            name = species_classes[pred_group][str(idx)]
            name = SPECIES_MERGE.get(name, name)
            results.append({"species": name, "prob": prob * group_conf})
    else:
        name = SPECIES_MERGE.get(GROUPS[pred_group][0], GROUPS[pred_group][0])
        results = [{"species": name, "prob": group_conf}]
    return results, pred_group, group_conf


# ─── Ver.4.0 / Ver.4.1 推論（埋め込み＋線形分類器、共通） ──────
def predict_embedding_model(model, clf, wav_path):
    emb_df = model.embed([wav_path], progress_bar=False) if hasattr(model, "embed") else None
    feature = emb_df.values.mean(axis=0).reshape(1, -1)
    probs = clf.predict_proba(feature)[0]
    classes = clf.classes_
    order = np.argsort(probs)[::-1][:TOP_K]
    return [{"species": classes[i], "prob": probs[i]} for i in order]


# ─── 結果表示の共通コンポーネント ───────────────────────────
def render_result_column(col, title, results, acc_multi=None, acc_single=None, extra_caption=None):
    with col:
        st.markdown(f"### {title}")
        if extra_caption:
            st.caption(extra_caption)
        if not results:
            st.error("判定失敗")
            return
        top = results[0]
        sp, conf = top["species"], float(top["prob"])
        st.markdown(f"**{sp}**")
        info = SPECIES_INFO.get(sp, {})
        if info:
            st.caption(f"*{info.get('latin','')}* / {info.get('en','')}")
        st.progress(min(conf, 1.0), text=f"確信度 {conf:.1%}")

        if sp in SINGLE_LOCATION_SPECIES:
            st.warning(f"⚠️ {sp}は単一調査地データのみ。他地点での信頼性は未検証。")
        if conf < 0.4:
            st.warning("確信度が低く、専門家確認を推奨します。")

        with st.expander("上位候補を見る"):
            for r in results:
                st.progress(min(float(r["prob"]), 1.0), text=f"{r['species']}  {r['prob']:.1%}")

        if acc_multi is not None:
            st.caption(f"場所ベース評価（複数地点種）: 精度{acc_multi:.1%}"
                       f"　/　単一地点種（参考値）: 精度{acc_single:.1%}")
        else:
            st.caption("場所ベース評価は未実施（既知の限界）")


# ─── UI ─────────────────────────────────────────────────
st.set_page_config(page_title="コウモリ音声識別 3方式比較", page_icon="🦇", layout="wide")

if COVER_IMAGE.exists():
    st.image(str(COVER_IMAGE), use_container_width=True)

st.title("🦇 日本産コウモリ音声識別 3方式比較アプリ Ver.4.2")
st.caption("Ver.3.2（階層分類CNN）／Ver.4.0（Google Perch転移学習）／"
           "Ver.4.1（BirdNET転移学習）を同一音声で同時実行し、横並びで比較します。")

st.warning(
    "**このアプリは3方式の比較検証用です。**  \n"
    "Ver.3.2は場所を考慮しない評価のみで、真の未知地点性能は未検証です。  \n"
    "Ver.4.0・Ver.4.1は場所ベース評価済み（複数地点17種で精度45〜50%程度）。  \n"
    "いずれも研究用の試作であり、確信度が低い場合や3方式で結果が割れる場合は、"
    "専門家による確認を強く推奨します。"
)

st.warning(
    "🔧 **修正済み・引き続き要注意（Ver.3.2）**：以前は`species_groups.py`の"
    "グループ定義に「ドーベントンコウモリ」が誤って含まれておらず、この種を"
    "原理的に判定不可能でした（2026年9月に修正・再学習済み）。"
    "現在は候補として出力可能になりましたが、学習データが依然4件と極端に少ないため、"
    "実際のテストでは正しく判定できていません（テスト2件とも誤判定）。"
    "ドーベントンコウモリの疑いがある録音は、3方式とも慎重に扱い、専門家判断を優先してください。"
)

with st.spinner("3つのモデルを読み込んでいます（初回は数十秒〜1分程度かかります）..."):
    ver32_models = load_ver32()
    ver40_perch, ver40_clf = load_ver40()
    ver41_birdnet, ver41_clf = load_ver41()

st.success("3モデルの準備が完了しました")
st.divider()

uploaded = st.file_uploader("WAVファイルをアップロード", type=["wav", "WAV"])

if uploaded is not None:
    audio_bytes = uploaded.read()
    st.audio(audio_bytes, format="audio/wav")

    with st.spinner("3方式で解析中..."):
        try:
            data, sr_orig = load_wav_bytes(audio_bytes)
            f_kHz, t, Sxx_dB = make_spectrogram(data, sr_orig)
            spec_img = spectrogram_to_pil(f_kHz, t, Sxx_dB)

            # Ver.3.2
            tensor = spectrogram_to_tensor(Sxx_dB)
            results_32, pred_group, group_conf = predict_ver32(ver32_models, tensor)

            # Ver.4.0 / Ver.4.1（共通のピッチシフト前処理）
            tmp_path = pitch_shift_to_tempfile(data, sr_orig)
            results_40 = predict_embedding_model(ver40_perch, ver40_clf, tmp_path)
            results_41 = predict_embedding_model(ver41_birdnet, ver41_clf, tmp_path)
        except Exception as e:
            st.error(f"解析エラー: {e}")
            st.stop()

    col1, col2 = st.columns(2)
    col1.metric("元サンプルレート", f"{sr_orig / 1000:.0f} kHz")
    col2.metric("録音時間", f"{len(data)/sr_orig:.2f} 秒")
    st.divider()

    # 3方式が一致しているかの簡易サマリー
    top_species = {results_32[0]["species"], results_40[0]["species"], results_41[0]["species"]}
    if len(top_species) == 1:
        st.success(f"✅ 3方式とも第1候補が一致：**{list(top_species)[0]}**")
    elif len(top_species) == 2:
        st.warning("⚠️ 3方式のうち2方式は一致、1方式が異なります。下記で詳細をご確認ください。")
    else:
        st.error("❌ 3方式の第1候補がすべて異なります。専門家による確認を強く推奨します。")

    c1, c2, c3 = st.columns(3)
    render_result_column(c1, "Ver.3.2（階層分類CNN）", results_32,
                          acc_multi=ACC["ver32"]["multi"], acc_single=ACC["ver32"]["single"],
                          extra_caption=f"音響グループ判定: {GROUP_LABELS.get(pred_group, pred_group)}"
                                        f"（確信度{group_conf:.1%}）")
    render_result_column(c2, "Ver.4.0（Perch転移学習）", results_40,
                          acc_multi=ACC["ver40"]["multi"], acc_single=ACC["ver40"]["single"])
    render_result_column(c3, "Ver.4.1（BirdNET転移学習）", results_41,
                          acc_multi=ACC["ver41"]["multi"], acc_single=ACC["ver41"]["single"])

    st.divider()
    st.subheader("スペクトログラム（10〜130 kHz、元の超音波録音）")
    st.image(spec_img, use_container_width=True)

    st.info(
        "**ご注意**：3方式は前提が異なります。Ver.3.2はスペクトログラム画像を直接CNNで分類、"
        "Ver.4.0/4.1は音声を音響埋め込みモデル（鳥類用に事前学習）に通してから線形分類器で判定します。"
        "  \nVer.4.0/4.1では、超音波をサンプルレート変換によるピッチシフト"
        "（タイム・エクスパンションと同義）で可聴域相当に変換してから入力しています。"
        "  \n学習データ: 3方式とも日本産22種（Ver.3.2は2,515録音、Ver.4.0/4.1は2,514録音・usableデータ全件）"
    )
