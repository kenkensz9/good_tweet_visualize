# good_tweet_visualize
良いツイートを取得し、可視化をするプログラムです。ツイートは外部のサーバーから取得しています。それ取り込みmain.pyでバックエンドとしてデータ処理をします。フロントエンドは./realtime_tweet_app/public2/index.htmlで処理しています。ミドルエンドは./realtime_tweet_app/server.jsでnode.jsを用いてwebsocketを使いながらフロントに、スクロールに合わせたりリアルタイムにデータ転送処理をしています。
