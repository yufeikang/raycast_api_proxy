# build stage
FROM python:3.10 AS builder

# install uv
RUN pip install uv

# copy files
COPY pyproject.toml README.md /project/
COPY raycast_proxy /project/raycast_proxy

# install dependencies and project
WORKDIR /project
RUN uv pip install -e "." --system

# run stage
FROM python:3.10-slim

# copy installed packages and project files from build stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /project/raycast_proxy /project/raycast_proxy
COPY scripts/entrypoint.sh /

EXPOSE 80

WORKDIR /project
ENV PYTHONPATH=/project/raycast_proxy

# set command/entrypoint, adapt to fit your needs
ENTRYPOINT ["/entrypoint.sh"]
