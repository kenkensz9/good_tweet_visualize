import pika
import json
import asyncio
import time
async def process_msg(msg):
    #msg=msg[0]
    print("received msg",msg)
    #print("遅延(秒)",time.time()-msg['send_raw_time'])
    #print("content",msg['content'])
    print()
def callback(ch, method, properties, body):
    msg = json.loads(body)
    asyncio.run(process_msg(msg))

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='tweets_queue')

# メッセージを受信して処理
channel.basic_consume(queue='tweets_queue', on_message_callback=callback, auto_ack=True)

print('Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
