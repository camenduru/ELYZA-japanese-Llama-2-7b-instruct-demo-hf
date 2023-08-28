---
title: ELYZA-japanese-Llama-2-7b-instruct-demo
emoji: ✨
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: 3.41.0
app_file: app.py
pinned: false
suggested_hardware: a10g-small
duplicated_from: elyza/ELYZA-japanese-Llama-2-7b-instruct-demo
---

# ELYZA-japanese-Llama-2-7b-instruct-demo
## 概要
- [ELYZA-japanese-Llama-2-7b](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b)は、[株式会社ELYZA](https://elyza.ai/) (以降「当社」と呼称) が[Llama2](https://ai.meta.com/llama/)をベースとして日本語能力を拡張するために事前学習を行ったモデルです。
- [ELYZA-japanese-Llama-2-7b-instruct](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b-instruct)は[ELYZA-japanese-Llama-2-7b](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b)を弊社独自のinstruction tuning用データセットで事後学習したモデルです。
    - 本デモではこのモデルが使われています。
- [ELYZA-japanese-Llama-2-7b-fast-instruct](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b-fast-instruct)は[ELYZA-japanese-Llama-2-7b](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b)に日本語語彙を追加した[ELYZA-japanese-Llama-2-7b-fast](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-7b-fast)を弊社独自のinstruction tuning用データセットで事後学習したモデルです。
    - このモデルを使ったデモは[こちら](https://huggingface.co/spaces/elyza/ELYZA-japanese-Llama-2-7b-fast-instruct-demo)です
- 詳細は[Blog記事](https://note.com/elyza/n/na405acaca130)を参照してください。
- 本デモではこちらの[Llama-2 7B Chat](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat)のデモをベースにさせていただきました。

## License
- Llama 2 is licensed under the LLAMA 2 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved.

## 免責事項
- 当社は、本デモについて、ユーザーの特定の目的に適合すること、期待する機能・正確性・有用性を有すること、出力データが完全性、正確性、有用性を有すること、ユーザーによる本サービスの利用がユーザーに適用のある法令等に適合すること、継続的に利用できること、及び不具合が生じないことについて、明示又は黙示を問わず何ら保証するものではありません。
- 当社は、本デモに関してユーザーが被った損害等につき、一切の責任を負わないものとし、ユーザーはあらかじめこれを承諾するものとします。
- 当社は、本デモを通じて、ユーザー又は第三者の個人情報を取得することを想定しておらず、ユーザーは、本デモに、ユーザー又は第三者の氏名その他の特定の個人を識別することができる情報等を入力等してはならないものとします。
- ユーザーは、当社が本デモ又は本デモに使用されているアルゴリズム等の改善・向上に使用することを許諾するものとします。

## 本デモで入力・出力されたデータの記録・利用に関して
- 本デモで入力・出力されたデータは当社にて記録させていただき、今後の本デモ又は本デモに使用されているアルゴリズム等の改善・向上に使用させていただく場合がございます。

## We are hiring!
- 当社 (株式会社ELYZA) に興味のある方、ぜひお話ししませんか？
- 機械学習エンジニア・インターン募集: https://open.talentio.com/r/1/c/elyza/homes/2507
- カジュアル面談はこちら: https://chillout.elyza.ai/elyza-japanese-llama2-7b
