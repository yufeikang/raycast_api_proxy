# Raycast AI Proxy

This is a simple proxy for the [Raycast AI](https://raycast.com/) app. It allows you to use the [Raycast AI](https://raycast.com/ai) app without subscribing. It is a simple proxy that forwards the requests to OpenAI's API and returns the response.

## Usage

### Installation in local

1. Clone the repository
2. Install the dependencies with `pip install -r requirements.txt`
3. Create a `.env` file with the following content:

```
OPENAI_API_KEY=<your openai api key>
```

4. Generate self-signed certificates with `./scripts/cert_gen.py -d bankend.raycast.com -o ./cert`
5. Run the server with `python3 main.py`

### Configuration

1. modify `/etc/host` to add the following line:

```
127.0.0.1 bankend.raycast.com
```

then you can use the Raycast AI app with the proxy.

2. trust the certificate in your system keychain

open the certificate in the `cert` folder and add it to the system keychain.
