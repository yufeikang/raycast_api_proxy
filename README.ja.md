# Raycast AI Proxy

これはシンプルな[Raycast AI](https://raycast.com/) APIプロキシです。サブスクリプションなしで[Raycast 
AI](https://raycast.com/ai)アプリを使用できるようにします。このプロキシは、RaycastからのリクエストをOpenAI APIに転送し、フォーマットを変換してリアルタイムで応答を返します。

[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md)

## 概要

### 対応モデル
複数のモデルは、対応する環境変数を設定することで同時に使用できます。

| プロバイダー | モデル名 | テスト状況 | 環境変数 |
| --- | --- | --- | --- |
| `openai` | gpt-3.5-turbo, gpt-4-turbo, gpt-4o | テスト済み | `OPENAI_API_KEY` |
| `azure openai` | - | テスト済み | `AZURE_OPENAI_API_KEY`, `AZURE_DEPLOYMENT_ID`, `OPENAI_AZURE_ENDPOINT` |
| `google` | gemini-pro, gemini-1.5-pro | テスト済み | `GOOGLE_API_KEY` |

### AIチャット

!(./assert/img/chat.jpeg)

### 翻訳

!(./assert/img/translate.jpg)

## 使用方法

### Dockerでのクイックスタート

1. 証明書の生成

```sh
pip3 install mitmproxy
python -c "$(curl -fsSL https://raw.githubusercontent.com/yufeikang/raycast_api_proxy/main/scripts/cert_gen.py)"  --domain backend.raycast.com  --out ./cert
```

2. サービスの起動

```sh
docker run --name raycast \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -p 443:443 \
    --dns 1.1.1.1 \
    -v $PWD/cert/:/data/cert \
    -e CERT_FILE=/data/cert/backend.raycast.com.cert.pem \
    -e CERT_KEY=/data/cert/backend.raycast.com.key.pem \
    -e LOG_LEVEL=INFO \
    -d \
    ghcr.io/yufeikang/raycast_api_proxy:main
```

3. AZURE OpenAI APIに切り替える

PythonでOpenAIとAzure OpenAIエンドポイントを切り替える方法については、以下を参照してください：[How to switch between OpenAI and Azure OpenAI endpoints with 
Python](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints)

```sh
docker run --name raycast \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e OPENAI_API_BASE=https://your-resource.openai.azure.com \
    -e OPENAI_API_VERSION=2023-05-15 \
    -e OPENAI_API_TYPE=azure \
    -e AZURE_DEPLOYMENT_ID=your-deployment-id \
    -p 443:443 \
    --dns 1.1.1.1 \
    -v $PWD/cert/:/data/cert \
    -e CERT_FILE=/data/cert/backend.raycast.com.cert.pem \
    -e CERT_KEY=/data/cert/backend.raycast.com.key.pem \
    -e LOG_LEVEL=INFO \
    -d \
    ghcr.io/yufeikang/raycast_api_proxy:main
```

4. Google Geminiのサポート（実験的機能）

> OpenAI APIと同時に使用可能で、対応する環境変数を設定するだけです

[Google API Key](https://makersuite.google.com/app/apikey)を取得し、`GOOGLE_API_KEY`としてエクスポートします。

```sh
# このリポジトリをクローンし、ディレクトリに移動
docker build -t raycast .
docker run --name raycast \
    -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
    -p 443:443 \
    --dns 1.1.1.1 \
    -v $PWD/cert/:/data/cert \
    -e CERT_FILE=/data/cert/backend.raycast.com.cert.pem \
    -e CERT_KEY=/data/cert/backend.raycast.com.key.pem \
    -e LOG_LEVEL=INFO \
    -d \
    raycast:latest
```

### ローカルインストール

1. このリポジトリをクローンしてください
2. `pdm install`を使用して依存関係をインストールします
3. 環境変数を設定します

```sh
export OPENAI_API_KEY=<your openai api key>
```

4. `./scripts/cert_gen.py --domain backend.raycast.com  --out ./cert`を使用して自己署名証明書を生成します
5. `python ./app/main.py`でサービスを起動します

### 設定

1. `/etc/hosts`を編集し、以下の行を追加します：

```
127.0.0.1 backend.raycast.com
::1 backend.raycast.com
```

この設定変更の目的は、`backend.raycast.com`を実際の`backend.raycast.com`の代わりにローカルホストにポイントすることです。DNSサーバーにこのレコードを追加することもできます。

2. システムキーチェーンに証明書の信頼を追加

`cert`フォルダ内のCA証明書を開き、システムキーチェーンに追加して信頼します。これは**必要**です。Raycast AI 
Proxyは自己署名証明書を使用しており、正常に動作するためには信頼される必要があります。

注意：

Apple Siliconを搭載したmacOSを使用している場合、CA証明書を手動で`Keychain Access`に追加するとアプリケーションが停止することがある場合、以下のコマンドを使って代替手段として追加できます：

(https://docs.mitmproxy.org/stable/concepts-certificates/#installing-the-mitmproxy-ca-certificate-manually)

```sh
sudo security add-trusted-cert -d -p ssl -p basic -k /Library/Keychains/System.keychain ~/.mitmproxy/mitmproxy-ca-cert.pem
```