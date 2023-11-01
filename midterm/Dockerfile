FROM python:3.11
ENV MODEL_PATH=/app/model.bin
ENV DV_PATH=/app/dv.bin
ENV TRAINING_DATA_PATH=/app/amazon_laptop_prices_v01.csv
WORKDIR /app

RUN pip install poetry

COPY poetry.lock pyproject.toml ./

RUN poetry config virtualenvs.create false
RUN poetry install --without dev

COPY predict.py dv.bin model.bin clean.py data ./
COPY /static ./static

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
