FROM python:3.10.8


WORKDIR /code

# requirements rarely change
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# will change frequently
COPY . /code
RUN pip install /code


ENTRYPOINT ["pytest", "-s", "/code"]