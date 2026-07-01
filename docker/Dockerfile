FROM python:3.11-slim

WORKDIR /app

COPY dist/ /tmp/dist/
RUN WHEEL=$(ls /tmp/dist/*.whl) && \
    pip install --no-cache-dir "${WHEEL}[web]" && \
    rm -rf /tmp/dist

EXPOSE 8080

CMD ["deeplc", "gui"]
