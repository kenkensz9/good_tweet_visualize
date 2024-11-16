import requests
import re
import sys
from openai import OpenAI
import openai
import os
#from google.colab import drive
from datetime import datetime, timedelta
import time
import os
import base64
import string
import csv

# OpenAI APIキーを設定
client = OpenAI()


def contains_keywords(text):
    # キーワードリスト
    global keywords
    # 各キーワードが文章に含まれているかをチェック
    for keyword in keywords:
        if keyword in text:
            return True  # キーワードが見つかった場合
    return False  # キーワードが見つからなかった場合
def gettweetinfo(tweet_div):
    tweet_menu = tweet_div .find("a", class_="Tweet_TweetMenu__44l_v")
    tweet = tweet_div.find("div", class_="Tweet_Tweet__bq4XS")
    if tweet:
        body_container = tweet.find("div", class_="Tweet_bodyContainer__n_Cs6")
        if body_container:
            tweet_info = body_container.find("div", class_="Tweet_info__5pNCA")
            #print(tweet_info)
            if tweet_info:
                # 著者名を取得
                tweet_author = tweet_info.find("p", class_="Tweet_author__vbXd2")
                if tweet_author:
                    #print("Author Name:", tweet_author.text)
                    pass
                # 表示名を取得
                tweet_author_name = tweet_info.find("span", class_="Tweet_authorName__V3waK")
                if tweet_author_name:
                    #print("Author Display Name:", tweet_author_name.text)
                    pass
                # 著者IDを取得
                tweet_author_id = tweet_info.find("a", class_="Tweet_authorID__B1U8c")
                match = re.search(idpattern, str(tweet_author_id))
                #print(match.group(1))
                if tweet_menu:
                  cl_params = tweet_menu.get("data-cl-params")
                  match2 = re.search(twidpattern, cl_params)
                return match.group(1), tweet_author_name.text,match2.group(1)

def remove_urls(text):
    # 正規表現を使ってURLを検出し削除
    clean_text = re.sub(r'http[s]?://\S+|www\.\S+|pic\.x\.com/\S+|x\.com/\S+|hoyme\.jp/\S+', '', text)
    return clean_text

def tensuka(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "あなたは文章を10点満点で、面白い度、不謹慎度、ロマン度、bot度、変人度、幻想度、頭いい度、現代メディア文化語彙知識依存度、自己言及度、詩的な雰囲気度、比喩と言葉遊び要素度、初見における文脈の理解しやすさ度を点数化するアシスタントです。また、最後に コメント: といったようにコメントを生成してください"
            },
            {
                "role": "user",
                "content": f"例：他の人なら抑え込めるだろうモノが自分が抑えられなくなっているのは感じる。このツイートが既にそう\n\n結果：\n面白い度：10\n不謹慎度：7\nロマン度：8\nbot文度：2\n変人度：8\n幻想度：6\n頭いい度：9\n現代メディア文化語彙知識依存度：0\n自己言及度：10\n詩的な雰囲気度：9\n比喩と言葉遊び要素度：4\n初見における文脈の理解しやすさ度：8\nコメント：狂気と知性がうまく混ざっている。"
            },
            {
                "role": "user",
                "content": f"例：{text}"
            }
        ],
        temperature=0,
        max_tokens=400,
        top_p=0.39,
        frequency_penalty=1.38,
        presence_penalty=1.12,
        #stop=["コメント："]
    )
    res = response.choices[0].message.content
    #print("resコメントとる",res)
    res = res.replace(':', '：')
    match = re.search(r'コメント： (.*)', res)
    comment = match.group(1)
    res = res.replace(' ', '')  # Remove all spaces
      # Replace all colons with full-width colons
    res = res.split("\n")
    res = [r.split("：")[1] if len(r.split("：")) > 1 else r for r in res]
    return (
        int(res[0]),  # 面白い度
        int(res[1]),  # 不謹慎度
        int(res[2]),  # ロマン度
        int(res[3]),  # bot度
        int(res[4]),  # 変人度
        int(res[5]),  # 幻想度
        int(res[6]),  # 頭いい度
        int(res[7]),  # 現代メディア文化語彙知識依存度
        int(res[8]),  # 自己言及度
        int(res[9]),  # 詩的な雰囲気度
        int(res[10]), # 言葉遊び要素度
        int(res[11]),
        comment
        # 意味の簡潔度
    )

def kanzyo(text):
  response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": text+"\n\nこのツイートに含まれている感情を、\n喜び 恍惚 平穏 \n期待 警戒 興味\n怒り 激怒  煩さ\n嫌悪 憎悪 退屈\n悲しみ 悲痛 憂い\n驚き 驚嘆 動揺\n恐れ 恐怖 心配\n信頼 感嘆 容認\n\nの中から挙げて"
        }
      ]
    }
  ],
  temperature=1,
  max_tokens=2048,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  response_format={
    "type": "text"
  }
)
  res = response.choices[0].message.content
  emotions_columns = [
    ["恍惚", "喜び", "平穏"],
    ["警戒", "期待", "興味"],
    ["激怒", "怒り", "煩さ"],
    ["憎悪", "嫌悪", "退屈"],
    ["悲痛", "悲しみ", "憂い"],
    ["驚嘆", "驚き", "動揺"],
    ["恐怖", "恐れ", "心配"],
    ["感嘆", "信頼", "容認"]
]
  detected_emotions = []

  for column in emotions_columns:
      for emotion in column:
          if emotion in res:
              detected_emotions.append(emotion)
              break  # 最初に見つかった感情を優先して次の列へ

  # 結果を出力
  if detected_emotions:
      print(f"検出された感情: {', '.join(detected_emotions)}")
  return detected_emotions


def ai_setumei_comment(text):
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": text+" \n\nこのツイートにおける趣旨の本質的なところを可視化できるように画像生成するための英語のプロンプトを生成して。ただし画像は生成しないで。\n\nまた、人間や背景や文化は自然なレベルで現代日本らしくして。また、前後関係や因果関係なども画像から分かるように画像に入れて。文章中に人間がいない場合は人間がいなくてもいい。その場合は概念を描いて。また、文章が長く画像を2つに分けた方が伝わる時は2つに分けて。その場合、最初が想像・過程で後ろが現実・現在の状況にして。また、それぞれにセクションを分けて簡単な説明文や独り言のような言葉を入れて。また、画像2では、画像1の人物等の特徴を省略しないで同じように説明して。またツイートの文末表現は特に取り入れて、ダイナミックに表現をして 。また、プロンプトは英語でキャプションは全て日本語で表示して。\n\nまた、出力は \n画像1のプロンプト(理想): \n画像1のキャプション： \n\n画像2のプロンプト(現実): \n画像1のキャプション： \nにして。"
          }
        ]
      },
    ],
    temperature=0,
    max_tokens=4096,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
      "type": "text"
    }
  )
  res = response.choices[0].message.content
  print("res画像",res)
  text = res.replace(':', '：') 
  prompt1_match = re.search(r"画像1のプロンプト\(理想\)：(.*?)(?:\n\n|画像1のキャプション)", text, re.DOTALL)
  prompt1 = prompt1_match.group(1).strip() if prompt1_match else "Not found"

  # 画像1のキャプションを取得
  caption1_match = re.search(r"画像1のキャプション：(.*?)(?:\n\n|画像2のプロンプト)", text, re.DOTALL)
  caption1 = caption1_match.group(1).strip() if caption1_match else "Not found"

  # 画像2のプロンプトを取得
  prompt2_match = re.search(r"画像2のプロンプト\(現実\)：(.*?)(?:\n\n|画像2のキャプション)", text, re.DOTALL)
  prompt2 = prompt2_match.group(1).strip() if prompt2_match else "Not found"

  # 画像2のキャプションを取得
  caption2_match = re.search(r"画像2のキャプション：(.*)", text, re.DOTALL)
  caption2 = caption2_match.group(1).strip() if caption2_match else "Not found"
  print("qq",prompt1, caption1, prompt2, caption2)
  return prompt1, caption1, prompt2, caption2
def check_and_mark(bun, omoshiroi, fukin, roman, mukisitu, henjin, gensou, atama, nankaido, jiko, shiteki, kotobawari, kanseki):
    # Check if the conditions meet for marking the tweet with NG or OK
    #kanseki 文脈の一般的度初見での理解しやすさ
    if fukin > 8 and nankaido < 3:
        return "〇不謹慎〇"
    if (mukisitu >= 6) and nankaido >= 5:
        return "bot感ある"
    if nankaido >= 4 and shiteki <= 4:
        return "オタクすぎ詩的ない"
    if nankaido >= 4 and roman <= 4:
        return "オタクすぎロマンない"
    if (atama <= 4) and (roman <= 6) and shiteki <= 5 and henjin>=5:
        return "頭悪い"
    if jiko <= 3 and henjin <= 4 and shiteki <= 5:
        return "他人事すぎ"
    if nankaido >= 8:
        return "文化依存すぎ"
    if shiteki <= 4:
        return "詩感なさすぎ"
    if kotobawari <= -1:
        return "言葉遊び感なさすぎ"
    if kanseki <= 4:
        return "文脈が不明瞭すぎ"
    if len(bun) >= 70:
        return "長すぎ"
    if (omoshiroi >= 8) or (fukin >= 5) or (roman >= 6) or (henjin >= 6) or (gensou >= 5) or (atama >= 6) or (shiteki >= 6) or (kotobawari >= 6):
        #writedrive(t[1],t[2],t[0],t[-1])
        return "〇"
    else:
        return "普通にNG"

# drive.mount('/content/drive')
def writedrive(dt,username, user_id, tweet_id, sentence,comment,p1,c1,p2,c2,kanzyolist,nitatext):
    name = "realtime_tweet.txt"
    #a = open("./src/" + name, "a", encoding="utf-8")
    a = open("./realtime_tweet_app/src/" + name, "a", encoding="utf-8")
    # 現在の時間を取得
    #current_time = datetime.now()
    current_time = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%fZ")
    current_time += timedelta(hours=9)
    current_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    writetext = (
        "\"" + str(username).replace(","," ").replace("\n", "").replace("\r", "") + "\"" + "," +
        "\"" + str(user_id).replace(","," ").replace("\n", "").replace("\r", "") + "\"" + "," +
        "\"" + str(tweet_id).replace(","," ").replace("\n", "").replace("\r", "") + "\"" + "," +  # ツイートIDを追加
        "\"" + current_time.replace(","," ").replace("\n", "").replace("\r", "") + "\"" + "," +
        "\"" + sentence.replace(","," ").replace("\n", "").replace("\r", "") + "\"" + "," +
        "\"" + comment.replace(","," ").replace("\n", "").replace("\r", "") + "\"" + "," +
        "\"" + p1.replace(","," ").replace("\n", "").replace("\r", "") + "\"" + "," +
        "\"" + c1.replace(","," ").replace("\n", "").replace("\r", "") + "\"" + "," +
        "\"" + p2.replace(","," ").replace("\n", "").replace("\r", "") + "\"" + "," +
        "\"" + c2.replace(","," ").replace("\n", "").replace("\r", "") + "\"" + "," +
        "\"" + " ".join(kanzyolist).replace(","," ").replace("\n", "").replace("\r", "") + "\"" + "," +
        "\"" + nitatext.replace(","," ").replace("\n", "").replace("\r", "") + "\""
    )
    # モードは0のみで、すべての情報を書き込む
    a.write("\n")
    a.write(writetext)

    a.close()  # ファイルを閉じる

def gazouseisei(prompt1,tweetnum,num):
    
  # Define the base URL for the API
  base_url = 'https://api.getimg.ai/v1'

  usemodel='realvis-xl-v4'
  # Endpoint for generating images
  generation_endpoint = '/flux-schnell/text-to-image'
  api_key=os.getenv('getimgapi')
  # Your API key (make sure to keep this confidential)
  # Define the headers with authentication
  headers = {
      'Authorization': f'Bearer {api_key}',
      'Content-Type': 'application/json',
  }

  # Define the number of images to generate
  number_of_images = 1

  # Define the JSON data for generating the image
  generation_data = {
      "model": "flux-schnell",  # Replace with your desired model
      "prompt": prompt1,
      "output_format": "jpeg",  # Specify JPEG format
      "width": 512,  # Your desired width
      "height": 512,  # Your desired height
      "steps":1,
  }

  # Create a directory to save generated images
  os.makedirs('generated', exist_ok=True)


  for i in range(number_of_images):
      # Make a POST request to generate the image
      try:
          response_generation = requests.post(f'{base_url}{generation_endpoint}', headers=headers, json=generation_data)

          # Check if the generation request was successful (status code 200)
          if response_generation.status_code == 200:
              # Parse the response JSON for the generated image
              generated_data = response_generation.json()
              
              # Decode the base64 image string
              generated_image_data = base64.b64decode(generated_data['image'])
              
              # Extract the first three words from the prompt
              prompt_words = generation_data['prompt'].split()[:3]
              
              # Generate a random string for uniqueness
              random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
              
              # Construct the filename with the first three words and random string
              #filename = '-'.join(prompt_words) + '-' + random_string + '-generated.jpg'
              generated_file_path = "./realtime_tweet_app/filesrc/gendimg/"+str(tweetnum) + '-' + str(num) + '.jpg'
              # Specify the file path for saving the generated image
              #generated_file_path = os.path.join('generated', filename)
              
              # Save the generated image to the specified file path
              with open(generated_file_path, 'wb') as image_file:
                  image_file.write(generated_image_data)
              
              print(f"Image {i+1} generated successfully and saved as '{generated_file_path}'")
          else:
              # Handle error cases for generation
              print(f'Error generating image {i+1}: {response_generation.status_code}')
              print(response_generation.text)
      except requests.exceptions.RequestException as e:
          # Handle connection errors or other exceptions here
          print(f'Error: {e}')

def vectorize_text(text):
    #print(f"ベクトル化するテキスト: {text}")  # デバッグ用
    client = OpenAI()
    #print("OpenAIクライアントを初期化しました。")  # デバッグ用
    response = client.embeddings.create(
        model="text-embedding-3-small",  # 使用するモデルを指定
        input=text
    )
    #print(f"APIからのレスポンス: {response}")  # デバッグ用
    embedding = response.data[0].embedding
    #print(f"生成されたベクトル: {embedding}")  # デバッグ用
    return embedding

def find_similar_tweets(new_tweet_vector, vector_list, tweetdata,top_n=11):
    #print(f"新しいツイートのベクトル: {new_tweet_vector}")  # デバッグ用
    similarities = []
    #print(f"ベクトルリストの長さ: {len(vector_list)}")  # デバッグ用
    for index, vector in enumerate(vector_list):
        #print(f"インデックス: {index}, 現在のベクトル: {vector}")  # デバッグ用
        similarity = cosine_similarity(new_tweet_vector, vector)  # コサイン類似度を計算する関数を呼び出す
        #print(f"インデックス: {index}, 類似度: {similarity}")  # デバッグ用
        similarities.append((tweetdata[index][1], similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    #print(f"ソートされた類似度: {similarities}")  # デバッグ用
    return similarities[:top_n][1:]

def load_vectors(file_path):
    global vector_list, tweet_data
    vector_list, tweet_data = [], []
    #print(f"ベクトルを読み込むファイル: {file_path}")  # デバッグ用
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            print("ファイルを開きました。")  # デバッグ用
            reader = csv.reader(f)
            for row in reader:
                #print(f"読み込んだ行: {row}")  # デバッグ用
                tweet_id, text, vector = row[0], row[1], eval(row[2])  # ベクトルをリストとして評価
                tweet_data.append((tweet_id, text))
                vector_list.append(vector)
                #print(f"追加されたツイートID: {tweet_id}, 本文: {text}, ベクトル: {vector}")  # デバッグ用
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {file_path}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    print(f"読み込まれたベクトル数: {len(vector_list)}")  # デバッグ用
    return vector_list, tweet_data

def save_vectors(file_path, tweet_id, text, vector):
    #print(f"保存するデータ: ツイートID: {tweet_id}, 本文: {text}, ベクトル: {vector}")  # デバッグ用
    with open(file_path, 'a', encoding='utf-8', newline='') as f:
        #print("ファイルを追記モードで開きました。")  # デバッグ用
        writer = csv.writer(f)
        writer.writerow([tweet_id, text, vector])  # 各行にツイートID、本文、ベクトルを保存
        #print(f"データがファイルに保存されました: {file_path}")  # デバッグ用

def cosine_similarity(vec1, vec2):
    #print(f"コサイン類似度を計算: vec1: {vec1}, vec2: {vec2}")  # デバッグ用
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    #print(f"ドット積: {dot_product}")  # デバッグ用
    norm_a = sum(a * a for a in vec1) ** 0.5
    #print(f"ベクトル1のノルム: {norm_a}")  # デバッグ用
    norm_b = sum(b * b for b in vec2) ** 0.5
    #print(f"ベクトル2のノルム: {norm_b}")  # デバッグ用
    similarity = dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0
    #print(f"計算されたコサイン類似度: {similarity}")  # デバッグ用
    return similarity

file_path='./src/vector.csv'
vector_list,tweet_data=load_vectors(file_path) 

import pika
import json
import asyncio
import time
tweetnum=0
import functools
import threading
from pika.exchange_type import ExchangeType

# ログ設定
#LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
#              '-35s %(lineno) -5d: %(message)s')
#LOGGER = logging.getLogger(__name__)

# ログの基本設定
#logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


# メッセージをACK（確認応答）する関数
def ack_message(ch, delivery_tag):
    print("ack_message")
    """`ch` はACKする際のpikaチャンネルインスタンスを渡す必要がある。
    """
    if ch.is_open:
        # チャンネルが開いている場合にACKを送信
        ch.basic_ack(delivery_tag)
    else:
        # チャンネルが閉じている場合、ACKが送れないためエラーハンドリング
        pass

def syori(msg):
    #print("received msg",msg,type(msg))
    #import pdb; pdb.set_trace() # 追加
    msg=json.loads(msg)
    try:
      msg=json.loads(msg)
    except:
        pass
    print("msg",msg)
    id,name,twid=msg['user_id'],msg['display_name'],msg['id']
    tweet_body = msg['text']
    dt=msg['datetime']
    print("(抽出情報→",id,name,twid,end="  ")
    print("(抽出ツイート→",tweet_body)
    try:        #print()    try:
       omoshiroi, fukin, roman, mukisitu, henjin, gensou, atama, nankaido, jiko, shiteki, kotobawari, kanseki, comment = tensuka(tweet_body)
    except Exception as e:
        print(f"2Error processing tweet: {e}")
        print("contents",tweet_body)
        print()
        return
    print("面白い度", omoshiroi, "不謹慎度", fukin, "ロマン度", roman, "無機質文度", mukisitu, "変人度", henjin, "幻想度", gensou, "頭いい度", atama, "文化依存度", nankaido, "自己言及度", jiko, "詩的な雰囲気度", shiteki, "言葉遊び要素度", kotobawari, "文脈の一般的度", kanseki)
    print(tweet_body)
    hyoka=check_and_mark(tweet_body, omoshiroi, fukin, roman, mukisitu, henjin, gensou, atama, nankaido, jiko, shiteki, kotobawari, kanseki)
    if hyoka=="〇":
        print("評価→",hyoka)
        prompt1, caption1, prompt2, caption2=ai_setumei_comment(tweet_body)
        print("prompt1",prompt1)
        print("caption1",caption1)
        print("prompt2",prompt2)
        print("caption2",caption2)
        gazouseisei(prompt1,twid,1)
        gazouseisei(prompt2,twid,2)  
        print()
        kanzyolist=kanzyo(tweet_body)
        new_tweet_vector = vectorize_text(tweet_body)
        save_vectors(file_path,twid,tweet_body,new_tweet_vector)
        vector_list.append(new_tweet_vector)
        tweet_data.append((twid, tweet_body))
        similar_tweets = find_similar_tweets(new_tweet_vector, vector_list,tweet_data)
        nitatext=""
        for s in similar_tweets:
            nitatext+=str(int(round(s[1]*100,0)))+"% "+s[0]+"<br>"
            #print(text)
            #print("類似ツイートインデックスと類似度:", similar_tweets)

        writedrive(dt,name,id,twid,tweet_body,comment,prompt1,caption1,prompt2,caption2,kanzyolist,nitatext)
    else:
        print("評価→",hyoka)
        print("-----------------------")
        #writedrive(t[1],t[2],t[0],t[-1])
    #twid,id,name,remove_urls(tweet_body.text)
        #print("回目処理を完了　待機中---------------------")          
        #time.sleep(random.randint(280, 490))
        #print("回目処理を完了　待機終了---------------------")

    #print("遅延(秒)",time.time()-msg['send_raw_time'])
    #print("content",msg['content'])
    print()

# メッセージを処理する関数
def do_work(ch, delivery_tag, body):
    global tweetnum
    tweetnum+=1
    print("tweetnum",tweetnum)
    try:
      #print(body)
      syori(body)
    except Exception as e:
        print(e)
        pass
    thread_id = threading.get_ident()  # スレッドIDの取得
    cb = functools.partial(ack_message, ch, delivery_tag)
    # ACKをスレッドセーフに送信するためのコールバック追加
    ch.connection.add_callback_threadsafe(cb)


# メッセージを受信した際のコールバック関数
def on_message(ch, method_frame, _header_frame, body, thrds):
    print("on_message")
    #thrds = args  # スレッドリストを引数として受け取る
    #print(thrds)
    delivery_tag = method_frame.delivery_tag
    # 新しいスレッドを作成し、メッセージを処理する
    t = threading.Thread(target=do_work, args=(ch, delivery_tag, body))
    t.start()
    thrds.append(t)  # スレッドをリストに追加して追跡する


# RabbitMQ接続設定
credentials = pika.PlainCredentials('test_user', 'test_password')  # テスト用のユーザー名とパスワード
# 心拍数を5秒に設定（長時間実行時に接続を維持するため）
parameters = pika.ConnectionParameters(
    'localhost', credentials=credentials, heartbeat=5)
connection = pika.BlockingConnection(parameters)  # ブロッキング接続の作成

channel = connection.channel()
# RabbitMQ Exchangeの宣言
channel.exchange_declare(
    exchange="test_tweets_queue",  # テスト用のキュー名
    exchange_type=ExchangeType.direct,
    passive=False,
    durable=True,
    auto_delete=False)

# Queueの宣言（メッセージを保持するキュー）
channel.queue_declare(queue="test_tweets_queue", auto_delete=False)  # テスト用のキュー名
# キューとExchangeのバインディング（ルーティングキーで指定）
channel.queue_bind(
    queue="test_tweets_queue", exchange="test_tweets_queue", routing_key="test_tweets_queue")  # テスト用のキュー名

# プレフェッチ数を1に設定（並列スレッド数を抑えるための設定）
channel.basic_qos(prefetch_count=1)

# スレッドリストの初期化
threads = []
# on_message関数を使用するための部分適用
on_message_callback = functools.partial(on_message, thrds=threads)
# メッセージ受信の開始
channel.basic_consume(on_message_callback=on_message_callback, queue='test_tweets_queue')

try:
    # メッセージの受信と処理の開始
    print("start")
    channel.start_consuming()
    print("end")
except KeyboardInterrupt:
    # キーボード割り込み時に停止
    channel.stop_consuming()

# すべてのスレッドが完了するまで待機
for thread in threads:
    thread.join()

# 接続を閉じる
connection.close()
