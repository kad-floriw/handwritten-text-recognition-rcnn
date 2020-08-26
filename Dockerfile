FROM tensorflow/tensorflow:1.15.2-py3

LABEL maintainer="Wim Florijn <wimflorijn@hotmail.com>"

ARG PORT=5000
ARG WORKERS=1
ARG TIMEOUT=30
ARG MAX_REQUESTS=1000

RUN apt-get update \
    && apt-get install -y libsm6 libxext6 libxrender-dev

COPY docker/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY src /app/src
COPY wsgi.py /app/wsgi.py
COPY weights /app/weights

WORKDIR /app

COPY docker/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENV PORT=$PORT
ENV WORKERS=$WORKERS
ENV TIMEOUT=$TIMEOUT
ENV MAX_REQUESTS=$MAX_REQUESTS

EXPOSE $PORT

ENTRYPOINT [ "/app/entrypoint.sh" ]
CMD [ "run" ]
