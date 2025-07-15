FROM python:3.13-rc-slim
WORKDIR /app/

# install binaries - ffmpeg (or avconv) needed by pydub (to be fast)
RUN apt-get update -y
RUN apt-get install ffmpeg gcc cargo -y

COPY ./.env /app/src/
COPY ./requirements.txt /app/

RUN pip install -r requirements.txt

COPY ./mypy.ini /app/
COPY ./src/*.py /app/src/
RUN mypy src --config-file /app/mypy.ini

WORKDIR /app/src/
ENTRYPOINT [ "python", "app.py" ]