import os
from predict import predict_upvotes
from fastapi import FastAPI

# Write the model version here or find some way to derive it from the model
# eg. from the model files name
model_version = "1"

# Set the log path.
# This should be a directory that is writable by the application.
# In a docker container, you can use /var/log/ as the directory.
# Mount this directory to a volume on the host machine to persist the logs.
log_dir_path = "/logs"
log_path = f"{log_dir_path}/V-{model_version}.log"

app = FastAPI()


@app.get("/ping")
def ping():
  return "ok"


@app.get("/version")
def version():
  return {"version": model_version}


@app.get("/logs")
def logs():
  return read_logs(log_path)

@app.post("/how_many_upvotes")
def how_many_upvotes(post):

    start_time = os.times().elapsed     # Start time for latency calculation
    prediction = predict_upvotes(post)  # Placeholder for actual prediction
    end_time = os.times().elapsed       # End time for latency calculation
    latency = (end_time - start_time) * 1000

    message = {}
    message["Latency"]    = latency
    message["Version"]    = model_version
    message["Timestamp"]  = end_time
    message["Input"]      = post.json()
    message["Prediction"] = prediction

    try:
        log_request(log_path, message.json())
    except Exception as e:
        pass

    return {"upvotes": prediction}


# Functions for logging and for predicting upvotes


##### Log The Request #####
def log_request(log_path, message):
  # print the message and then write it to the log
  pass


##### Read The Logs #####
def read_logs(log_path):
  # read the logs from the log_path
  pass
