FROM svizor/zoomcamp-model:3.10.12-slim
ENV MODEL_PATH=/app/model2.bin
ENV DV_PATH=/app/dv.bin
WORKDIR /app
RUN pip install pipenv
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy
COPY main.py ./
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "main:app"]
EXPOSE 9696