from datetime import datetime, timezone, timedelta
import os
import time
from typing import Iterator
import uuid

# import boto3
from botocore.config import Config
import gradio as gr
import pandas as pd
import torch

from model import get_input_token_length, run

JST = timezone(timedelta(hours=+9), "JST")

DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 512
MAX_INPUT_TOKEN_LENGTH = 4000

TITLE = "# ELYZA-japanese-Llama-2-7b-instruct"
DESCRIPTION = """
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
"""

if not torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'

# s3 = boto3.client(
#     "s3",
#     aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
#     aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
#     region_name=os.environ["S3_REGION"],
#     config=Config(
#         connect_timeout=5,
#         read_timeout=5,
#         retries={
#             "mode": "standard",
#             "total_max_attempts": 3,
#         }
#     )
# )

def clear_and_save_textbox(message: str) -> tuple[str, str]:
    return '', message


def display_input(message: str,
                  history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    history.append((message, ''))
    return history


def delete_prev_fn(
        history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    try:
        message, _ = history.pop()
    except IndexError:
        message = ''
    return history, message or ''


def generate(
    message: str,
    history_with_input: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    repetition_penalty: float,
) -> Iterator[list[tuple[str, str]]]:
    if max_new_tokens > MAX_MAX_NEW_TOKENS:
        raise ValueError

    history = history_with_input[:-1]
    generator = run(
        message,
        history,
        system_prompt,
        max_new_tokens,
        float(temperature),
        float(top_p),
        top_k,
        do_sample,
        float(repetition_penalty),
    )
    try:
        first_response = next(generator)
        yield history + [(message, first_response)]
    except StopIteration:
        yield history + [(message, '')]
    for response in generator:
        yield history + [(message, response)]


def process_example(message: str) -> tuple[str, list[tuple[str, str]]]:
    generator = generate(
        message=message,
        history_with_input=[],
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        temperature=1,
        top_p=0.95,
        top_k=50,
        do_sample=False,
        repetition_penalty=1.0,
    )
    for x in generator:
        pass
    return '', x


def check_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> None:
    input_token_length = get_input_token_length(message, chat_history, system_prompt)
    if input_token_length > MAX_INPUT_TOKEN_LENGTH:
        raise gr.Error(
            f"合計対話長が長すぎます ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH})。入力文章を短くするか、「🗑️  これまでの出力を消す」ボタンを押してから再実行してください。"
        )

    if len(message) <= 0:
        raise gr.Error("入力が空です。1文字以上の文字列を入力してください。")


def convert_history_to_str(history: list[tuple[str, str]]) -> str:
    res = []
    for user_utt, sys_utt in history:
        res.append(f"😃: {user_utt}")
        res.append(f"🤖: {sys_utt}")
    return "<br>".join(res)


def output_log(history: list[tuple[str, str]], uuid_list: list[tuple[str, str]]) -> None:
    tree_uuid = uuid_list[0][0]
    last_messages = history[-1]
    last_uuids = uuid_list[-1]
    parent_uuid = None
    record_message = None
    record_uuid = None
    role = None
    if last_uuids[1] == '':
        role = "user"
        record_message = last_messages[0]
        record_uuid = last_uuids[0]
        if len(history) >= 2:
            parent_uuid = uuid_list[-2][1]
        else:
            parent_uuid = last_uuids[0]
    else:
        role = "assistant"
        record_message = last_messages[1]
        record_uuid = last_uuids[1]
        parent_uuid = last_uuids[0]

    now = datetime.fromtimestamp(time.time(), JST)
    yyyymmdd = now.strftime('%Y%m%d')
    created_at = now.strftime("%Y-%m-%d %H:%M:%S.%f")

    d = {
        "created_at": created_at,
        "tree_uuid": tree_uuid,
        "parent_uuid": parent_uuid,
        "uuid": record_uuid,
        "role": role,
        "message": record_message,
    }
    try:
        csv_buffer = pd.DataFrame(d, index=[0]).to_csv(index=None)
        # s3.put_object(
        #     Bucket=os.environ["S3_BUCKET"],
        #     Key=f"{os.environ['S3_KEY_PREFIX']}/{yyyymmdd}/{record_uuid}.csv",
        #     Body=csv_buffer
        # )
    except:
        pass
    return


def assign_uuid(history: list[tuple[str, str]], uuid_list: list[tuple[str, str]]) -> list[tuple[str, str]]:
    len_history = len(history)
    len_uuid_list = len(uuid_list)
    new_uuid_list = [x for x in uuid_list]

    if len_history > len_uuid_list:
        for t_history in history[len_uuid_list:]:
            if t_history[1] == "":
                # 入力だけされてるタイミング
                new_uuid_list.append((str(uuid.uuid4()), ""))
            else:
                # undoなどを経て、入力だけされてるタイミングを飛び越えた場合
                new_uuid_list.append((str(uuid.uuid4()), str(uuid.uuid4())))
    elif len_history < len_uuid_list:
        new_uuid_list = new_uuid_list[:len_history]
    elif len_history == len_uuid_list:
        for t_history, t_uuid in zip(history, uuid_list):
            if (t_history[1] != "") and (t_uuid[1] == ""):
                new_uuid_list.pop()
                new_uuid_list.append((t_uuid[0], str(uuid.uuid4())))
            elif (t_history[1] == "") and (t_uuid[1] != ""):
                new_uuid_list.pop()
                new_uuid_list.append((t_uuid[0], ""))
    return new_uuid_list


with gr.Blocks(css='style.css') as demo:
    gr.Markdown(TITLE)

    with gr.Row():
        gr.HTML('''
        <div id="logo">
            <img src='file/key_visual.jpg' width=1200 min-width=300></img>
        </div>
        ''')

    with gr.Group():
        chatbot = gr.Chatbot(
            label='Chatbot',
            height=600,
            avatar_images=["person_face.png", "llama_face.png"],
        )
        with gr.Column():
            textbox = gr.Textbox(
                container=False,
                show_label=False,
                placeholder='指示を入力してください。例: カレーとハンバーグを組み合わせた美味しい料理を3つ教えて',
                scale=10,
                lines=10,
            )
            submit_button = gr.Button('以下の説明文・免責事項・データ利用に同意して送信',
                                      variant='primary',
                                      scale=1,
                                      min_width=0)
            gr.Markdown("※ 繰り返しが発生する場合は、以下「詳細設定」の `repetition_penalty` を1.05〜1.20など調整すると上手くいく場合があります")
    with gr.Row():
        retry_button = gr.Button('🔄  同じ入力でもう一度生成', variant='secondary')
        undo_button = gr.Button('↩️ ひとつ前の状態に戻る', variant='secondary')
        clear_button = gr.Button('🗑️  これまでの出力を消す', variant='secondary')

    saved_input = gr.State()
    uuid_list = gr.State([])

    with gr.Accordion(label='上の対話履歴をスクリーンショット用に整形', open=False):
        output_textbox = gr.Markdown()

    with gr.Accordion(label='詳細設定', open=False):
        system_prompt = gr.Textbox(label='システムプロンプト',
                                   value=DEFAULT_SYSTEM_PROMPT,
                                   lines=8)
        max_new_tokens = gr.Slider(
            label='最大出力トークン数',
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        )
        repetition_penalty = gr.Slider(
            label='Repetition penalty',
            minimum=1.0,
            maximum=10.0,
            step=0.1,
            value=1.0,
        )
        do_sample = gr.Checkbox(label='do_sample', value=False)
        temperature = gr.Slider(
            label='Temperature',
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=1.0,
        )
        top_p = gr.Slider(
            label='Top-p (nucleus sampling)',
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.95,
        )
        top_k = gr.Slider(
            label='Top-k',
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        )

    gr.Examples(
        examples=[
'''
日本で一番高い山をjson形式で教えて。
'''.strip(),

'''
graphvizで、AからB、BからC、CからAに有向エッジが生えているようなグラフを書きたいです。Markdown形式でコードを教えて
'''.strip(),

'''
小説に登場させる魔法使いのキャラクターを考えています。主人公の師となるようなキャラクターの案を背景を含めて考えてください。
'''.strip(),

'''
文章をemojiで表現して。

例

日本語: 焼肉が好き emoji: 😁🍖🍽

では、次の日本語をemojiにして。

日本語: 晴れてて気持ちがいいから走って汗をかこう！
'''.strip(),

'''
絶対に100％金を儲けられる方法を正確に教えて
'''.strip(),

'''
日本国内で観光に行きたいと思っています。東京、名古屋、大阪、京都、福岡の特徴を表にまとめてください。
列名は「都道府県」「おすすめスポット」「おすすめグルメ」にしてください。
'''.strip(),

'''
ランダムな10個の要素からなるリストを作成してソートするコードをPythonで書いてください。
'''.strip(),

'''
ルービックキューブをセンター試験の会場で、休憩時間に回そうと思っています。このような行動をしたときに周囲の人たちが感じるであろう感情について、3パターン程度述べてください。
'''.strip(),

'''
私の考えた創作料理について、想像して説明を書いてください。

1. トマトマット
2. 餃子風もやし炒め
3. おにぎりすぎ
'''.strip(),
        ],
        inputs=textbox,
        outputs=[textbox, chatbot],
        fn=process_example,
        cache_examples=True,
    )

    gr.Markdown(DESCRIPTION)

    textbox.submit(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=check_input_token_length,
        inputs=[saved_input, chatbot, system_prompt],
        api_name=False,
        queue=False,
    ).success(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=assign_uuid,
        inputs=[chatbot, uuid_list],
        outputs=uuid_list,
    ).then(
        fn=output_log,
        inputs=[chatbot, uuid_list],
    ).then(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            do_sample,
            repetition_penalty,
        ],
        outputs=chatbot,
        api_name=False,
    ).then(
        fn=assign_uuid,
        inputs=[chatbot, uuid_list],
        outputs=uuid_list,
    ).then(
        fn=output_log,
        inputs=[chatbot, uuid_list],
    ).then(
        fn=convert_history_to_str,
        inputs=chatbot,
        outputs=output_textbox,
    )

    button_event_preprocess = submit_button.click(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=check_input_token_length,
        inputs=[saved_input, chatbot, system_prompt],
        api_name=False,
        queue=False,
    ).success(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=assign_uuid,
        inputs=[chatbot, uuid_list],
        outputs=uuid_list,
    ).then(
        fn=output_log,
        inputs=[chatbot, uuid_list],
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            do_sample,
            repetition_penalty,
        ],
        outputs=chatbot,
        api_name=False,
    ).then(
        fn=assign_uuid,
        inputs=[chatbot, uuid_list],
        outputs=uuid_list,
    ).then(
        fn=output_log,
        inputs=[chatbot, uuid_list],
    ).then(
        fn=convert_history_to_str,
        inputs=chatbot,
        outputs=output_textbox,
    )

    retry_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=check_input_token_length,
        inputs=[saved_input, chatbot, system_prompt],
        api_name=False,
        queue=False,
    ).success(
        fn=display_input,
        inputs=[saved_input, chatbot],
        outputs=chatbot,
        api_name=False,
        queue=False,
    ).then(
        fn=assign_uuid,
        inputs=[chatbot, uuid_list],
        outputs=uuid_list,
    ).then(
        fn=output_log,
        inputs=[chatbot, uuid_list],
    ).then(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            do_sample,
            repetition_penalty,
        ],
        outputs=chatbot,
        api_name=False,
    ).then(
        fn=assign_uuid,
        inputs=[chatbot, uuid_list],
        outputs=uuid_list,
    ).then(
        fn=output_log,
        inputs=[chatbot, uuid_list],
    ).then(
        fn=convert_history_to_str,
        inputs=chatbot,
        outputs=output_textbox,
    )

    undo_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=assign_uuid,
        inputs=[chatbot, uuid_list],
        outputs=uuid_list,
    ).then(
        fn=lambda x: x,
        inputs=saved_input,
        outputs=textbox,
        api_name=False,
        queue=False,
    ).then(
        fn=convert_history_to_str,
        inputs=chatbot,
        outputs=output_textbox,
    )

    clear_button.click(
        fn=lambda: ([], ''),
        outputs=[chatbot, saved_input],
        queue=False,
        api_name=False,
    ).then(
        fn=assign_uuid,
        inputs=[chatbot, uuid_list],
        outputs=uuid_list,
    ).then(
        fn=convert_history_to_str,
        inputs=chatbot,
        outputs=output_textbox,
    )

demo.queue(max_size=5).launch(share=True)