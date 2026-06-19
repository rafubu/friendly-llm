FROM python:3.11-slim

WORKDIR /app

COPY . /app/packages/signaling/
RUN pip install -e /app/packages/signaling

RUN mkdir -p /data

EXPOSE 9876

ENV LITERT_SIGNALING_DB_PATH=/data/signaling.db

CMD ["litert-signaling", "serve", "--host", "0.0.0.0", "--port", "9876"]
