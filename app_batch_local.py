# -*- coding: utf-8 -*-
"""
日本産コウモリ音声 一括分類アプリ Ver.4.3（ローカル実行専用）
自動録音機のフォルダを指定すると、全WAVを Perch・BirdNET（22種＋ノイズ）で判定し、Excel一覧表を作る。

起動: start_batch_app.bat をダブルクリック（または venv_perch で streamlit run batch_app.py）
"""
import pathlib
import pandas as pd
import streamlit as st

import batch_core as core

st.set_page_config(page_title="コウモリ一括分類 Ver.4.3", page_icon="🦇", layout="wide")
st.title("🦇 日本産コウモリ音声 一括分類 Ver.4.3")
st.caption("自動録音機のフォルダを丸ごと判定し、Excel一覧表を作成します（Perch・BirdNETの2方式、22種＋ノイズ）。")
st.warning(
    "研究用の試作です。未知の調査地での種同定精度は Perch 49.6%・BirdNET 45.8%（複数地点17種）。"
    "種名は「候補」として扱い、「要確認」の行は必ずスペクトログラムで確認してください。  \n"
    "ノイズクラスは虫・雑音・川の音（6調査地49録音）と無音区間のみで学習しており、"
    "雨・風・鳥・人工音などは十分に学習していません。"
)


@st.cache_resource
def get_models():
    return core.load_models()


folder = st.text_input("対象フォルダのパス", placeholder=r"例: D:\録音\2026沢渡\SD01")
recursive = st.checkbox("サブフォルダも含める", value=True)

if folder:
    folder_p = pathlib.Path(folder.strip().strip('"'))
    if not folder_p.is_dir():
        st.error("フォルダが見つかりません")
        st.stop()
    files = core.list_wavs(folder_p, recursive)
    st.write(f"WAVファイル: **{len(files)} 件**（目安：1件あたり約2〜3秒 → 約{len(files) * 2.5 / 60:.0f}分）")

    out_default = str(core.default_output_path(folder_p))
    existing = sorted(folder_p.glob("コウモリ一括判定_*_途中経過.csv"))
    if existing:
        st.info(f"途中経過が見つかりました：{existing[-1].name}  \n同じ出力名で開始すると続きから再開します。")
        out_default = str(existing[-1]).replace("_途中経過.csv", ".xlsx")
    out_path = st.text_input("出力Excelのパス", value=out_default)

    if st.button("判定開始", type="primary", disabled=len(files) == 0):
        with st.spinner("モデルを読み込んでいます（初回は1分程度）..."):
            models = get_models()
        bar = st.progress(0.0, text="開始...")

        def show(done, total, elapsed):
            rate = elapsed / max(done, 1)
            bar.progress(done / total, text=f"{done}/{total} 件完了（残り約{(total - done) * rate / 60:.0f}分）")

        xlsx, df = core.run(folder_p, out_path, recursive, models=models, on_progress=show)
        bar.progress(1.0, text="完了")
        st.success(f"Excelを保存しました：{xlsx}")

        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("判定別の件数")
            st.dataframe(df["総合判定"].value_counts().rename("件数"))
            st.metric("要確認", f"{(df['要確認'] == '要確認').sum()} 件")
        with c2:
            st.subheader("一覧（先頭100件）")
            st.dataframe(df[["ファイル名", "録音日時", "総合判定", "要確認", "確認理由",
                             "Perch第1候補", "Perch確信度", "BirdNET第1候補", "BirdNET確信度"]].head(100))
        with open(xlsx, "rb") as f:
            st.download_button("Excelをダウンロード", f, file_name=pathlib.Path(xlsx).name)
