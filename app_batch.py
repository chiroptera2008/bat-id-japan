# -*- coding: utf-8 -*-
"""
日本産コウモリ音声 一括判定 Web版 Ver.4.3（Streamlit Cloud用）
複数のWAV、またはフォルダごと圧縮したZIPをアップロード → Perch・BirdNET（22種＋ノイズ）で判定 → Excelをダウンロード。
処理本体はローカル版と共通の batch_core.py。大量（数百件以上）のデータはローカル配布版を使う。

GitHubリポジトリ直下に app_batch.py / batch_core.py / ver43_batch/*.joblib を置く。
"""
import datetime as dt
import pathlib
import tempfile

import streamlit as st

import batch_core as core

MAX_FILES = 300   # Cloudの処理時間・メモリを考え、1回あたりの上限を設ける


# ─── パスワード認証（Ver.4.2と同じ方式、パスワードはsecretsで管理） ─────────
def check_password():
    if st.session_state.get("authenticated"):
        return True
    st.title("日本産コウモリ音声 一括判定 Ver.4.3")
    pw = st.text_input("パスワードを入力してください", type="password")
    if pw:
        if pw == st.secrets.get("APP_PASSWORD", ""):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("パスワードが違います")
    st.stop()


st.set_page_config(page_title="コウモリ一括判定 Ver.4.3", page_icon="🦇", layout="wide")
check_password()


@st.cache_resource
def get_models():
    return core.load_models()


st.title("🦇 日本産コウモリ音声 一括判定 Ver.4.3")
st.caption("複数のWAVファイル（またはフォルダごと圧縮したZIP）を、Perch・BirdNETの2方式で"
           "コウモリ22種＋ノイズに判定し、Excel一覧表にまとめます。")

st.warning(
    "**研究用の試作です。** 未知の調査地での種同定精度は Perch 49.6%・BirdNET 45.8%（複数地点17種）。"
    "種名は「候補」として扱い、「要確認」の行は必ずスペクトログラムで確認してください。  \n"
    "ノイズは虫・雑音・川の音（6調査地49録音）と無音区間のみで学習しており、"
    "雨・風・鳥・人工音などは十分に学習していません。"
)

with st.expander("使い方と、Web版の制限"):
    st.markdown(
        f"""
- **ZIPでのアップロードをおすすめします。** 録音フォルダを右クリック →「圧縮先」→「ZIPファイル」で作れます。
  ZIPなら録音日時（ファイルの日時）とフォルダ構成が保たれます。
- WAVを直接まとめて選んだ場合、ファイル名に `20200915_063713` のような日時が無いと、録音日時は「不明」になります。
- 1回に処理できるのは **{MAX_FILES}件まで**、アップロード容量は合計500MBまでです（500kHz・3秒の録音で約3MB/件）。
  処理には1件あたり数秒かかります。処理中はこのページを閉じないでください。
- それより多いデータは、ローカル配布版（フォルダを直接指定・中断再開可）をご利用ください。
"""
    )

uploads = st.file_uploader("WAVまたはZIPファイル", type=["wav", "zip"], accept_multiple_files=True)

if uploads and st.button("判定開始", type="primary"):
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="batid_web_"))
    with st.spinner("ファイルを展開しています..."):
        core.save_uploads(uploads, workdir)
    files = core.list_wavs(workdir)
    if not files:
        st.error("WAVファイルが見つかりませんでした")
        st.stop()
    if len(files) > MAX_FILES:
        st.error(f"WAVファイルが{len(files)}件あります。Web版では1回{MAX_FILES}件までです。分けてアップロードしてください。")
        st.stop()

    with st.spinner("モデルを読み込んでいます（初回は1分程度）..."):
        models = get_models()

    bar = st.progress(0.0, text=f"{len(files)}件の判定を開始...")

    def show(done, total, elapsed):
        rate = elapsed / max(done, 1)
        bar.progress(done / total, text=f"{done}/{total} 件完了（残り約{(total - done) * rate / 60:.0f}分）")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    xlsx = workdir.parent / f"コウモリ一括判定_{stamp}.xlsx"
    # 個別アップロードのWAVは更新日時が失われるため、batch_core側で録音日時「不明」になる
    xlsx, df = core.run(workdir, xlsx, models=models, on_progress=show)
    bar.progress(1.0, text="完了")
    st.session_state["result"] = (xlsx.name, xlsx.read_bytes(), df)

if "result" in st.session_state:
    name, data, df = st.session_state["result"]
    st.success(f"判定が完了しました（{len(df)}件）")
    st.download_button("Excelをダウンロード", data, file_name=name, type="primary",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("判定別の件数")
        st.dataframe(df["総合判定"].value_counts().rename("件数"))
        st.metric("要確認", f"{(df['要確認'] == '要確認').sum()} 件")
    with c2:
        st.subheader("一覧（先頭100件）")
        st.dataframe(df[["ファイル名", "フォルダ", "録音日時", "総合判定", "要確認", "確認理由",
                         "Perch第1候補", "Perch確信度", "BirdNET第1候補", "BirdNET確信度"]].head(100))
