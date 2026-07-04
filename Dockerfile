From python:3.13.5
WORKDIR /app

ENV PYTHONDONTWRITEByTECODE =1
ENV PYTHONBUFFERED=1

copy requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000
EXPOSE 8501

CMD ["bash","start.sh"]
