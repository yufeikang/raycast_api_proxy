# build stage
FROM mitmproxy/mitmproxy:10.4.1 AS builder

# install PDM
RUN pip install -U pip setuptools wheel
RUN pip install pdm

# Copy project files
COPY pyproject.toml pdm.lock README.md /project/
COPY scripts /project/scripts

# Install project dependencies
WORKDIR /project
COPY pdm.lock .
RUN mkdir __pypackages__ && pdm install --prod --no-editable

# Generate self-signed certificates in a temporary directory
RUN mkdir /temp_cert && pdm run /project/scripts/cert_gen.py -d backend.raycast.com -o /temp_cert

# run stage
FROM python:3.11.9-slim

# Install wget
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Retrieve packages from build stage
ENV PYTHONPATH=/project/pkgs
COPY --from=builder /project/__pypackages__/3.11/lib /project/pkgs
RUN mkdir /temp_cert
COPY --from=builder /temp_cert /temp_cert/cert
COPY app /project/app
COPY scripts/entrypoint.sh /

EXPOSE 80 443

WORKDIR /project
# Set command/entrypoint, adapt to fit your needs
ENTRYPOINT ["sh", "/entrypoint.sh"]
