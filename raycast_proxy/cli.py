"""Command line interface for raycast proxy server."""

import logging
import sys
from pathlib import Path

import click
import uvicorn

from raycast_proxy.cert_gen import generate_certificates

FORMAT = "%(asctime)-15s %(threadName)s %(filename)-15s:%(lineno)d %(levelname)-8s: %(message)s"


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", help="Port to bind to, default to 80, 443 if SSL enabled")
@click.option("--ssl/--no-ssl", default=False, help="Enable SSL")
@click.option(
    "--cert-file", help="Path to SSL certificate file (auto-generated if not provided)"
)
@click.option(
    "--key-file", help="Path to SSL key file (auto-generated if not provided)"
)
@click.option(
    "--domain", default="backend.raycast.com", help="Domain for SSL certificate"
)
@click.option("--log-level", default="INFO", help="Logging level")
def run(
    host: str,
    port: int,
    ssl: bool,
    cert_file: str,
    key_file: str,
    domain: str,
    log_level: str,
):
    """Start the raycast proxy server."""
    logging.basicConfig(format=FORMAT)
    logging.getLogger("raycast_proxy").setLevel(getattr(logging, log_level.upper()))

    ssl_cert_path = None
    ssl_key_path = None

    ssl_cert_path = None
    ssl_key_path = None

    if not port:
        port = 443 if ssl else 80

    if ssl:
        if cert_file and key_file:
            ssl_cert_path = Path(cert_file)
            ssl_key_path = Path(key_file)
        else:
            # Auto-generate certificates
            cert_dir = Path.cwd() / "cert"
            click.echo(f"Generating SSL certificates for {domain} in {cert_dir}")
            try:
                ssl_cert_path, ssl_key_path = generate_certificates(
                    domains=[domain],
                    out_dir=cert_dir,
                )
                click.echo("SSL certificates generated successfully")
            except Exception as e:
                click.echo(f"Failed to generate SSL certificates: {e}", err=True)
                sys.exit(1)

    click.echo(f"Starting server on {host}:{port}")
    if ssl:
        click.echo(f"SSL enabled with cert: {ssl_cert_path}, key: {ssl_key_path}")

    uvicorn.run(
        "raycast_proxy.main:app",
        host=host,
        port=port,
        ssl_certfile=ssl_cert_path,
        ssl_keyfile=ssl_key_path,
    )


def main():
    """Main entry point for the CLI."""
    run()


if __name__ == "__main__":
    main()
