# mmd

おんJ・おーぷん2ちゃんねる発のキャラの MMD モデル（PMX）を、Python で手続き的に作るリポジトリ。

| キャラ | ファイル | 身長 |
| --- | --- | --- |
| 束音ロゼ | `models/tabane_roze/tabane_roze.pmx` | 165cm |
| 蓄音キリコ | `models/chikune_kiriko/chikune_kiriko.pmx` | 158cm |
| フェリス | `models/feris/feris.pmx` | 150cm（仮） |

## 作り直す

```bash
pip install numpy pillow
python build.py                 # 全員
python build.py feris           # 1人だけ
```

形・色は `mmdgen/characters/*.py` を書き換える。共通の素体（ボーン・体型・5本指の手・顔・髪の部品）は `mmdgen/body.py`。

## 中身

- ボーン：全ての親／センター／グルーブ／上半身・上半身2／首／頭／両目・目／肩・腕・ひじ・手首／5本指（親指０〜２・人指〜小指１〜３）／下半身／足・ひざ・足首・つま先／足ＩＫ・つま先ＩＫ。一般的な MMD モーション（手指モーション含む）がそのまま当たる名前にしてある。
- モーフ：まばたき・笑い・ウィンク・ウィンク右・あ・い・う・え・お・にこり・怒り・困る（母音モーフでは口内の歯・舌が見え、あごも連動して下がる）
- テクスチャ：瞳（虹彩・ハイライト・瞳孔）・白目・まつ毛・眉・鼻筋・チーク・口（リップライン・上の歯／八重歯・舌・口内）・髪・衣装模様などを手続き生成して `models/<name>/textures/` に出力。
- ないもの：腕捩り・手捩りボーン。

## 確認用ビューア

```bash
python serve.py
```

ブラウザで http://localhost:8765/viewer/ が開き、three.js の MMDLoader で表示する（`python -m http.server 8765` でもよい）。

- 上のボタンでキャラを切り替える。「歩く」「走る」「ジャンプ」「ダンス」で動かし、髪やスカートの揺れ・服の突き抜けを見る。「速さ」でゆっくりにできる
- 「物理」のチェックを外すと髪やスカートが揺れなくなる
- `viewer/?m=feris&motion=run&at=0.4` のように開くと、その動きの 0.4 秒目で止まる（見比べや記録用）

## 顔を自分で調整する

ビューアと同じように開いた http://localhost:8765/viewer/tune.html （公開ページの viewer/tune.html でもよい）で、
左のスライダー（目の大きさ・縦の長さ・間隔・高さ・傾き、瞳の幅・高さ・上下・瞳孔・光、まつ毛の太さと明るさ、白目の影、瞳の色、顔の幅、頬の張り）を動かすと、
右の正面と斜めの顔がその場で変わる（ブラウザの中で頂点と目のテクスチャを動かすだけなので待ち時間がない）。上のボタンでキャラを切り替える。

「保存」を押したとき、`python serve.py` で開いていれば `mmdgen/characters/face/<キー>.json` に書いて正確に作り直す
（プレビューは近似なので、目の高さを変えたときの鼻・口の位置などは作り直して初めて正しくなる）。
`serve.py` がなければ `<キー>.json` がダウンロードされるので、`mmdgen/characters/face/` に置いて `python build.py` を実行する。
寸法は作るときに書き出す `models/<キー>/face.json` から読む。

ブラウザなしで正面の顔を PNG にするなら `python -m mmdgen.render feris out.png bare`（`bare` は髪と服を消して輪郭と目を見る、`face` は髪あり）。
関数 `render()` は `yaw`（斜め・横から）と `clay=True`（色を消して凹凸だけ）も使える。

頭の肌の形（顔・あご・後頭部・首の付け根）は、既製の MMD モデル「うか」の頭を `python -m mmdgen.head_scan` で一周輪切りにした表 `mmdgen/head_shape.py` から作る。
測り直すときは「うか」を手元に置く（リポジトリには入れない）：

```bash
git clone --depth 1 https://github.com/mmdagent-ex/uka.git models/uka_ref
python -m mmdgen.head_scan
```

顔の幅（頬の丸い萌え顔）・目の大きさと間隔・瞳／白目／まつ毛の塗り方は、音街ウナ公式 MMD モデル（SynthesizerV2 版、
https://otomachiuna.jp/download/download-detail/ ）を手本に合わせている。再配布禁止なので `models/una_ref/model.pmx` に手元で展開して比べるだけにする。

頭の形の表は CG アバター「うか」（CC-BY 4.0）から作っている：CG-CA Uka (c) 2023-2024 by Nagoya Institute of Technology, Moonshot R&D Goal 1 Avatar Symbiotic Society

## 利用規約について

二次創作の非公式モデル。各キャラの公式規約に従うこと。

- 束音ロゼ：[公式サイト](https://tabaneroze.ninja-web.net/)の[利用規約](https://tabaneroze.ninja-web.net/terms-of-use.html)
- 蓄音キリコ：[公式サイト](https://suzuhete.wixsite.com/home)の利用規約
- フェリス：[おんJwiki](https://w.atwiki.jp/openj3/pages/36.html)

どちらの規約も、収益のない利用と広告収益の範囲なら報告なしで使える。それ以外の収益化は公式への個別連絡が要る。自作発言・なりすまし・公序良俗に反する利用は禁止。3D モデル化と MMD には規約で触れていないので、気になる場合は公式に確認すること。

## VRM からの変換

Neural4D などで作った VRM は `blender/vrm_to_pmx.py` で PMX にできる（VRM アドオンと mmd_tools 入りの Blender を使う）。
骨を MMD の標準名にし、A ポーズ化・足 IK・表情モーフ（まばたき・あいうえお など）を付ける。
VRM のスプリングボーン（髪など）には、体の当たり判定と一緒に剛体・ジョイントを付けて揺れるようにする（`mmdgen/pmx_physics.py`）。

```
blender -b --python blender/vrm_to_pmx.py -- 入力.vrm models/xxx_n4d/xxx_n4d.pmx "モデル名"
```

Neural4D の出力は元の素材の再配布条件がはっきりしないので、`models/*_n4d/` はリポジトリに入れない。
