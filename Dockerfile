FROM python:3.10
RUN pip3 install fastapi uvicorn pydantic tensorflow pandas requests torch pickle transformers re

CMD [ "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "15400"]