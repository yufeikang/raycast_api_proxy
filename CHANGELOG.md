# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-02-27

### Added

- CLI interface for starting the server (`raycast-proxy[cli]` command)
- Automatic SSL certificate generation when running in HTTPS mode
- Command-line options for customizing server configuration:
  - SSL certificate domain customization
  - Custom certificate/key file paths
  - Host and port configuration
  - Logging level control
- Optional CLI dependencies install: Install from GitHub Release assets with `uv pip install <release-wheel-url>[cli]`

### Changed

- Renamed package from `app` to `raycast_proxy` for better package naming
- Restructured project for standard Python packaging
- Moved certificate generation from scripts to integrated CLI feature

### Dependencies

- Core dependencies remain minimal
- CLI features (uvicorn, cryptography, mitmproxy) moved to optional `cli` extras
- Upgraded all dependencies to latest compatible versions

## [0.1.0] - Initial Release

### Added

- Initial implementation of Raycast API proxy
- Support for multiple AI providers:
  - OpenAI
  - Anthropic
  - Google Gemini
  - DeepSeek
- Basic FastAPI server implementation
- Docker support
- Configuration via YAML files
