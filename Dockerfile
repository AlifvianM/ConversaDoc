FROM python:3.10

EXPOSE 8501

WORKDIR /usr/src/app

# dont write pyc files
# dont buffer to stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements_2.txt /usr/src/app/requirements_2.txt

# dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements_2.txt \
    && rm -rf /root/.cache/pip

COPY ./app /usr/src/app