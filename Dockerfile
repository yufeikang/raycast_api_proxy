# build stage
FROM python:3.10 AS builder

# install PDM and six
RUN pip install -U pip setuptools wheel
RUN pip install pdm
RUN pip install six

# copy files
COPY pyproject.toml pdm.lock README.md /project/
COPY scripts /project/scripts

# install dependencies and project into the local packages directory
WORKDIR /project
RUN mkdir __pypackages__ && pdm install --prod --no-lock --no-editable

# install cryptography for certificate generation
RUN pdm add cryptography

# generate self-signed certificates
RUN python ./scripts/cert_gen.py --domain backend.raycast.com --out ./cert

# run stage
FROM python:3.10-slim

# install wget
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# retrieve packages from build stage
ENV PYTHONPATH=/project/pkgs
COPY --from=builder /project/__pypackages__/3.10/lib /project/pkgs
COPY --from=builder /project/cert /project/cert
COPY app /project/app
COPY scripts/entrypoint.sh /

EXPOSE 80 443

WORKDIR /project
# set command/entrypoint, adapt to fit your needs
ENTRYPOINT ["sh", "/entrypoint.sh"]
