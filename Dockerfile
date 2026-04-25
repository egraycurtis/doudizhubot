FROM python:3.9-slim

WORKDIR /usr/src/app
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

COPY requirements-inference.txt ./
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY action_space.py cards.py filtered_options.py self_play.py turn_info.py ./
COPY inference.py serve.py ./
COPY models/transformer/transformer0.keras models/transformer/transformer1.keras models/transformer/transformer2.keras ./models/transformer/

EXPOSE 8080

CMD ["python", "serve.py"]
