# build stage
FROM python:3.9 AS builder

# install PDM
RUN pip install -U pip setuptools wheel
RUN pip install pdm

# copy files
COPY pyproject.toml pdm.lock README.md /project/

# install dependencies and project into the local packages directory
WORKDIR /project
RUN mkdir __pypackages__ && pdm install --prod --no-lock --no-editable

# run stage
FROM python:3.9

# retrieve packages from build stage
ENV PYTHONPATH=/project/pkgs
COPY --from=builder /project/__pypackages__/3.9/lib /project/pkgs
COPY app /project/app
COPY scripts/entrypoint.sh /

EXPOSE 80

WORKDIR /project/app
# set command/entrypoint, adapt to fit your needs
ENTRYPOINT sh /entrypoint.sh
