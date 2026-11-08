# mmd

おんJ・おーぷん2ちゃんねる発のキャラの MMD モデル（PMX）を、Python で手続き的に作るリポジトリ。

| キャラ | ファイル | 身長 |
| --- | --- | --- |
| 束音ロゼ | `models/tabane_roze/tabane_roze.pmx` | 165cm |
| 蓄音キリコ | `models/chikune_kiriko/chikune_kiriko.pmx` | 158cm |
| フェリス | `models/feris/feris.pmx` | 150cm（仮） |

## 作り直す

```bash
pip install numpy
python build.py                 # 全員
python build.py feris           # 1人だけ
```

形・色は `mmdgen/characters/*.py` を書き換える。共通の素体（ボーン・体型・顔・髪の部品）は `mmdgen/body.py`。

## 中身

- ボーン：全ての親／センター／グルーブ／上半身・上半身2／首／頭／両目・目／肩・腕・ひじ・手首／下半身／足・ひざ・足首・つま先／足ＩＫ・つま先ＩＫ。一般的な MMD モーションがそのまま当たる名前にしてある。
- モーフ：まばたき・笑い・ウィンク・ウィンク右・あ・い・う・え・お
- テクスチャなし（材質の色だけ）。目や口は顔の表面に貼った板。
- ないもの：指ボーン（手はミトン型）、捩りボーン、物理（剛体・ジョイント）。髪やスカートは揺れない。

## 確認用ビューア

```bash
python -m http.server 8765
```

http://localhost:8765/viewer/ で three.js の MMDLoader を使って表示する。「ポーズ確認」でボーンを動かし、ウェイトの具合を見られる。

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
