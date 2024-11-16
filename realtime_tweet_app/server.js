// server.js
const express = require('express');
const fs = require('fs');
const WebSocket = require('ws');
const app = express();
const https = require('https');
const port = 3001;
const path = require('path');
const cors = require('cors'); 
app.use(cors()); // CORSを有効にする

app.use(express.static('public'));
app.use(express.static('filesrc')); // publicフォルダ内の静的ファイルを提供
app.use('/good_tweet_kasika/realtime_tweet_app/public', express.static(path.join(__dirname, 'good_tweet_kasika/realtime_tweet_app/public')));
app.use('/good_tweet_kasika/realtime_tweet_app/filesrc', express.static(path.join(__dirname, 'good_tweet_kasika/realtime_tweet_app/filesrc')));

const options = {
    key: fs.readFileSync('/etc/letsencrypt/live/kenkensz9.com/privkey.pem'), // プライベートキーのパス
    cert: fs.readFileSync('/etc/letsencrypt/live/kenkensz9.com/fullchain.pem') // 証明書のパス
};

const filePath = './src/realtime_tweet.txt';

const server = https.createServer(options, app);
const wss = new WebSocket.Server({ server }); // WebSocketサーバーをHTTPSサーバーに統合

server.listen(port, () => {
    console.log(`Server is running at https://localhost:${port}`);
});

// 最後の行のインデックスを保存する変数
let lastLineIndex = -1; 

wss.on('connection', (ws) => {
    console.log('Client connected');

    ws.on('message', (message) => {
        const msg = typeof message == 'string' ? message : message.toString();
        console.log(`Received: ${msg}`);

        if (msg == '初回データをリクエスト') {
            // 初回データを送信
            fs.readFile(filePath, 'utf8', (err, data) => {
                if (err) {
                    console.error('Error reading file:', err);
                    return;
                }
                
                const tweets = data.split('\n').map(line => line.split(',')); // 各行をカンマで分割
                const reversedTweets = tweets.reverse(); // ツイートを逆順にする
                lastLineIndex = reversedTweets.length - 1; // 最後の行のインデックスを更新
                ws.send(JSON.stringify({ type: 'initial', data: reversedTweets })); // 逆順の全ツイートを送信
            });
        } else if (msg.startsWith('新しいツイートをリクエスト')) {
            // 新しいツイートを送信
            fs.readFile(filePath, 'utf8', (err, data) => {
                if (err) {
                    console.error('Error reading file:', err);
                    return;
                }
                
                const tweets = data.split('\n').map(line => line.split(',')); // 各行をカンマで分割

                // lastLineIndexよりも大きいインデックスを持つツイートをフィルタリング
                const newTweets = tweets.slice(lastLineIndex + 1); // lastLineIndex以降の新しい行を取得
                if (newTweets.length > 0) {
                    // 新しいツイートを逆順にして送信
                    const reversedNewTweets = newTweets.reverse(); // 逆順にする
                    ws.send(JSON.stringify({ type: 'new', data: reversedNewTweets })); // 新しいツイートを送信

                    // 最後の行のインデックスを更新
                    lastLineIndex += newTweets.length; // 新しいツイートの数だけインデックスを更新
                } else {
                    console.log("新しいツイートはありません。");
                }
            });
        }
    });

    ws.on('error', (error) => {
        console.error('WebSocket error:', error);
    });

    ws.on('close', () => {
        console.log('Client disconnected');
    });
});

// ファイルの変更を監視
fs.watch(filePath, (eventType, filename) => {
    if (eventType === 'change') {
        console.log(`${filename} has been changed.`);

        fs.readFile(filePath, 'utf8', (err, data) => {
            if (err) {
                console.error('Error reading file:', err);
                return;
            }

            const tweets = data.split('\n').map(line => line.split(',')); // 各行をカンマで分割
            const newTweets = tweets.slice(lastLineIndex + 1); // lastLineIndex以降の新しい行を取得
            if (newTweets.length > 0) {
                // 差分をデバッグ用に表示
                console.log("新しいツイートが見つかりました:", newTweets);

                // 最後の行のインデックスを更新
                lastLineIndex += newTweets.length; // 新しいツイートの数だけインデックスを更新
                wss.clients.forEach((client) => {
                    if (client.readyState === WebSocket.OPEN) {
                        client.send(JSON.stringify({ type: 'new', data: newTweets })); // 新しいツイートを送信
                    }
                });
            } else {
                console.log("新しいツイートはありません。");
            }
        });
    }
});
