FROM python:3.6.4
WORKDIR /mortgages/ml_api
ADD ./packages/ml_api /mortgages/ml_api
RUN pip install --upgrade pip \
&& pip install -r /mortgages/ml_api/requirements.txt
EXPOSE 5000
CMD python run.py