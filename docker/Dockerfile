FROM python:3.11-slim AS builder
RUN pip install uv
WORKDIR /build
COPY . .
RUN uv build --wheel --no-sources --out-dir /dist

FROM python:3.11-slim
COPY --from=builder /dist/*.whl /tmp/
RUN WHEEL=$(ls /tmp/*.whl) && \
    pip install --no-cache-dir "${WHEEL}[web]" && \
    rm "${WHEEL}"
EXPOSE 8080
CMD ["deeplc", "gui"]
