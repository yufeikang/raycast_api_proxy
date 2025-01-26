# Raycast AI Proxy

This is a simple [Raycast AI](https://raycast.com/) API proxy. It allows you to use the [Raycast AI](https://raycast.com/ai) application without subscribing. The proxy converts requests from Raycast into a format to send to AI model providers (e.g., OpenAI), and then converts the responses back into Raycast’s format.

[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md)

## Introduction

This project uses a man-in-the-middle approach to intercept and forward Raycast requests to various AI APIs, then returns the responses after reformatting them for Raycast. It primarily maps:

- `GET /api/v1/me`: Modifies the flag indicating user support for AI.  
- `POST /api/v1/translations`: Translation interface.  
- `POST /api/v1/ai/chat_completions`: Common AI interface.  
- `GET /api/v1/ai/models`: AI model list interface.

### How It Works (Man-in-the-Middle)

1. Modify DNS or `/etc/hosts` to point `backend.raycast.com` to this proxy instead of the official server.  
2. The proxy receives the HTTPS requests from Raycast.  
3. A self-signed certificate is used to decrypt and forward these requests to the configured AI endpoints (e.g., OpenAI, Anthropic).  
4. The responses are re-encrypted and returned to Raycast.  

Because Raycast and its API communicate via HTTPS, you need to trust the self-signed certificate for this interception to work. More details on man-in-the-middle proxies can be found at [mitmproxy documentation](https://docs.mitmproxy.org/stable/concepts-howmitmproxyworks/).

### YAML-Based Model Configuration

Environment variables are becoming deprecated. Now you can define multiple models in `config.yml`, allowing providers to coexist:

```yaml
models:
  - provider_name: "openai"
    api_type: "openai"
    params:
      api_key: "sk-xxxx"
      allow_model_patterns:
        - "gpt-\\d+"
  - provider_name: "azure openai"
    api_type: "openai"
    params:
      api_key: "xxxxxx"
      base_url: "https://your-resource.openai.azure.com"
      # ...
  - provider_name: "google"
    api_type: "gemini"
    params:
      api_key: "xxxxxx"
  - provider_name: "anthropic"
    api_type: "anthropic"
    params:
      api_key: "sk-ant-xxx"
  - provider_name: "deepseek"
    api_type: "openai"   # openai-compatible
    params:
      api_key: "sk-deepseek-xxx"
default_model: "gpt-4"
```

Each provider entry specifies the provider name, API type, and parameters.

- `provider_name`: The provider name. used for identification.
- `api_type`: The API type. For example, `openai`, `gemini`, or `anthropic`.
- `params`: `base_url`, `api_key`, and other parameters required by the provider.

Supported providers:
you can combine multiple models, Common options include:

| Provider | Model | Test Status | Image Generation |
| --- | --- | --- | --- |
| `openai` | **from api** | Tested | Supported |
| `azure openai` | Same as above | Tested | Supported |
| `google` | **from api** | Tested | Not supported |
| `anthropic` | claude-3-sonnet, claude-3-opus, claude-3-5-opus | Tested | Not supported |
| `deepseek` | **from api** | Tested | Not supported |
| `ollama` | **from api** | Tested | Not Supported |


Refer to the `config.yml.example` file for more details.

### Ai chat

![ai chat](./assert/img/chat.jpeg)

### Translate

![translate](./assert/img/translate.jpg)

### Image Generation

Only OpenAI API supports image generation.

## Usage

### Installation and Configuration

#### 1. Generate Certificate

```sh
pip3 install mitmproxy
python -c "$(curl -fsSL https://raw.githubusercontent.com/yufeikang/raycast_api_proxy/main/scripts/cert_gen.py)"  --domain backend.raycast.com  --out ./cert
```

Or

Clone this repository and run:

```sh
pdm run cert_gen
```

#### 2. Add Certificate to System Keychain

Open the CA certificate in the `cert` folder and add it to the system keychain and trust it.
This is **mandatory**, as the Raycast AI proxy uses a self-signed certificate, and it must be trusted to work correctly.

Note:

When using on macOS with Apple Silicon, if you encounter application hanging issues when manually adding the CA certificate to the "Keychain Access", you can use the following command in
the terminal as an alternative:

(<https://docs.mitmproxy.org/stable/concepts-certificates/#installing-the-mitmproxy-ca-certificate-manually>)

```shell
sudo security add-trusted-cert -d -p ssl -p basic -k /Library/Keychains/System.keychain ~/.mitmproxy/mitmproxy-ca-cert.pem
```

#### 3. Modify `/etc/host` to add the following lines

```
127.0.0.1 backend.raycast.com
::1 backend.raycast.com
```

The purpose of this modification is to redirect `backend.raycast.com` to the local machine, rather than the real `backend.raycast.com`. You can also add this record in your DNS server.

Alternatively, you can add this record to your DNS server. The ultimate goal is to make `backend.raycast.com` point to the address where this project is deployed. The `127.0.0.1` can be
replaced with your deployment address. If you deploy this project in the cloud or in your local network, you can point this address to your deployment address.

#### 4. Launch the service

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

You can also deploy this service in the cloud or your local network, as long as your Raycast can access this address.

**Then, restart Raycast, and you should be able to use it.**

### Advanced Configuration

#### 1. Using Azure OpenAI API

Refer to [How to switch between OpenAI and Azure OpenAI endpoints with Python](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints).

Simply modify the corresponding environment variables.

```sh
docker run --name raycast \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e OPENAI_BASE_URL=https://your-resource.openai.azure.com \
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

#### 2. Google Gemini API

> Can be used together with the OpenAI API by setting the corresponding environment variables.

Obtain your [Google API Key](https://makersuite.google.com/app/apikey) and export it as `GOOGLE_API_KEY`.

```sh
# git clone this repo and cd to it
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

#### 3. Local Manual Run

1. Clone this repository
2. Install dependencies using `pdm install`
3. Create environment variables

```
export OPENAI_API_KEY=<your openai api key>
```

4. Generate self-signed certificate using `./scripts/cert_gen.py --domain backend.raycast.com  --out ./cert`
5. Start the service using `python ./app/main.py`

#### 4. Local Development

Since you might have modified the local DNS, developing locally might lead to DNS loops. To avoid this, use Docker during local development and start the development environment by
specifying the DNS.

Reference:

```sh
sh ./local_docker.sh
```

#### 5. Using Custom Mapping

You can refer to the `custom_mapping.yml.example` file in the project directory to customize the modifications to some interface responses.

```yaml
"api/v1/me/trial_status":
  get:
    response:
      body:
        # json path replace
        "$.trial_limits.commands_limit": 30
```

For example, the above configuration will replace `$.trial_limits.commands_limit` in the response body of the `GET api/v1/me/trial_status` interface with `30`. The
`$.trial_limits.commands_limit` is a [JSON path](https://goessner.net/articles/JsonPath/).

Currently, only response body replacements are supported.

#### 6. Multi-User Shared Service

If you want to allow multiple users to share this service or you deploy the service on the public internet, you need to restrict which users can access the service. You can use the
`ALLOWED_USERS` environment variable to restrict which users can access the service.

```env
ALLOWED_USERS="xx@example.com,yy@example.com"
```

The email addresses are the Raycast user email addresses, separated by commas.

### Notes

1. DNS Designation
Due to the presence of GFW (Great Firewall of China), if you use this in mainland China, you might need to designate a domestic DNS server. Otherwise, domain names might not resolve
correctly. For instance: `--dns 223.5.5.5`.

2. DNS Not Taking Effect
Sometimes on macOS, modifying the `/etc/hosts` file does not take effect immediately. There’s no known solution to this yet. Sometimes restarting Raycast helps, or modifying the
`/etc/hosts` file again might work.

### Roadmap

- [ ] Support web search
- [ ] Support more AI models
- [ ] Improve project structure
