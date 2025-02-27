"""Certificate generation utilities."""

import os
from pathlib import Path
from typing import List

from cryptography.hazmat.primitives import serialization
from mitmproxy.certs import CertStore


def generate_certificates(
    domains: List[str], out_dir: Path, org: str = "mitmproxy"
) -> tuple[Path, Path]:
    """Generate SSL certificates for the given domains.

    Args:
        domains: List of domains to generate certificates for
        out_dir: Output directory for certificates
        org: Organization name for the certificate

    Returns:
        Tuple of (cert_path, key_path)
    """
    domain = domains[0]  # Use first domain as primary
    ca_dir = Path.home() / ".mitmproxy"

    certstore = CertStore.from_store(str(ca_dir), "mitmproxy", 2048)
    cert_entry = certstore.get_cert(domain, domains, org)
    cert = cert_entry.cert
    pkey = cert_entry.privatekey

    out_dir.mkdir(parents=True, exist_ok=True)

    cert_path = out_dir / f"{domain}.cert.pem"
    key_path = out_dir / f"{domain}.key.pem"
    ca_path = out_dir / "ca.cert.pem"

    # Write certificate
    with open(cert_path, "wb") as f:
        f.write(cert.to_pem())

    # Write private key
    with open(key_path, "wb") as f:
        buf = pkey.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        f.write(buf)

    # Write CA certificate
    with open(ca_path, "wb") as f:
        f.write(certstore.get_cert("mitmproxy", [], org).cert.to_pem())

    return cert_path, key_path
