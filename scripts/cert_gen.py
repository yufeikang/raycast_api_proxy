#! /usr/bin/env python3
# author: @yufeikang
# github: https://gist.github.com/yufeikang/7a4fee101d68ef2ac017db57205dd6bd
# This script is used to generate a certificate by mitmproxy
# Usage: cert_gen.py -d example.com -o ./certs
#

import argparse
import os

import OpenSSL
from cryptography.hazmat.primitives import serialization
from mitmproxy.certs import CertStore

parser = argparse.ArgumentParser(description="生成SSL/TSL证书")
parser.add_argument(
    "--domain",
    "-d",
    dest="domain",
    required=True,
    nargs="+",
    help="<Required> domain Set",
)
parser.add_argument(
    "--out", "-o", dest="out_dir", type=str, default="./", help="output dir"
)

parser.add_argument(
    "--org", dest="org", type=str, default="mitmporxy", help="openssl organization name"
)

args = parser.parse_args()

domains = args.domain
out_dir = args.out_dir
org = args.org

print("domains: {}, org:{}".format(domains, org))

domain = domains[0]

CA_DIR = os.path.expanduser("~/.mitmproxy")


certstore = CertStore.from_store(CA_DIR, "mitmproxy", 2048)

cert_entry = certstore.get_cert(domain, domains, org)
cert = cert_entry.cert
pkey = cert_entry.privatekey

with open(os.path.join(out_dir, "{}.cert.pem".format(domain)), "wb") as f:
    f.write(cert.to_pem())

with open(os.path.join(out_dir, "{}.key.pem".format(domain)), "wb") as f:
    buf = pkey.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    f.write(buf)

print("Success!!")
