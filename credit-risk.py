# main.py
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json

app = FastAPI()

class PredictionInput(BaseModel):
    # Define the input parameters required for making predictions
    pct_tl_open_L6M :float
    pct_tl_closed_L6M :float
    Tot_TL_closed_L12M :int
    pct_tl_closed_L12M :float
    Tot_Missed_Pmnt :int
    CC_TL :int
    Home_TL :int
    PL_TL :int
    Secured_TL :int
    Unsecured_TL :int
    Other_TL :int
    Age_Oldest_TL :int
    Age_Newest_TL :int
    time_since_recent_payment :int
    max_recent_level_of_deliq :int
    num_deliq_6_12mts :int
    num_times_60p_dpd :int
    num_std_12mts  :int
    num_sub :int
    num_sub_6mts :int
    num_sub_12mts :int
    num_dbt :int
    num_dbt_12mts :int
    num_lss :int
    recent_level_of_deliq :int
    CC_enq_L12m :int
    PL_enq_L12m :int
    time_since_recent_enq :int
    enq_L3m :int
    NETMONTHLYINCOME :int
    Time_With_Curr_Empr :int
    CC_Flag :int
    PL_Flag :int
    pct_PL_enq_L6m_of_ever :float
    pct_CC_enq_L6m_of_ever :float
    HL_Flag :int
    GL_Flag :int
    MARITALSTATUS :object
    EDUCATION :int
    GENDER :object
    last_prod_enq2 :object 
    first_prod_enq2 :object
    
    
model_path = "pipeline.joblib"
pipeline =load(model_path)
    
    
data_path='df.joblib'
data=load(data_path)

@app.get("/")
def home():
    return "Working fine"

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Extract features from input_data and make predictions using the loaded model
    features = [input_data.pct_tl_open_L6M, 
                input_data.pct_tl_closed_L6M, 
                input_data.Tot_TL_closed_L12M,
                input_data.pct_tl_closed_L12M,
                input_data.Tot_Missed_Pmnt, 
                input_data.CC_TL, 
                input_data.Home_TL, 
                input_data.PL_TL,
                input_data.Secured_TL, 
                input_data.Unsecured_TL, 
                input_data.Other_TL, 
                input_data.Age_Oldest_TL,
                input_data.Age_Newest_TL, 
                input_data.time_since_recent_payment,
                input_data.max_recent_level_of_deliq, 
                input_data.num_deliq_6_12mts, 
                input_data.num_times_60p_dpd,
                input_data.num_std_12mts, 
                input_data.num_sub, 
                input_data.num_sub_6mts, 
                input_data.num_sub_12mts, 
                input_data.num_dbt,
                input_data.num_dbt_12mts, 
                input_data.num_lss, 
                input_data.recent_level_of_deliq,
                input_data.CC_enq_L12m,
                input_data.PL_enq_L12m, 
                input_data.time_since_recent_enq, 
                input_data.enq_L3m, 
                input_data.NETMONTHLYINCOME,
                input_data.Time_With_Curr_Empr,
                input_data.CC_Flag, 
                input_data.PL_Flag, 
                input_data.pct_PL_enq_L6m_of_ever,
                input_data.pct_CC_enq_L6m_of_ever, 
                input_data.HL_Flag, 
                input_data.GL_Flag, 
                input_data.MARITALSTATUS,
                input_data.EDUCATION, 
                input_data.GENDER, 
                input_data.last_prod_enq2, 
                input_data.first_prod_enq2
                ]
    
    df=pd.DataFrame(data=[features],
        #[0.677, 0.0, 0, 0.0, 2, 0, 0, 1, 2, 1, 0, 18, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 46, 1, 150000, 154, 0, 1, 1.0, 0.0, 0, 0, 'Single', 3, 'F', 'PL', 'others']],
                       columns=data.columns[:-1])
    print(df)
    prediction = pipeline.predict(df)[0]
    # Return the prediction
    appr_flag_dict={
        'P1':  0,
        'P2' : 1,
        'P3' : 2,
        'P4' : 3
    }
    
    for key,value in appr_flag_dict.items():
        if prediction==value:
            print(key)
            return {'prediction': key}

 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# CMD: gunicorn -w 4 -k uvicorn.workers.UvicornWorker app_gunicorn:app

# Uvicorn is a lightweight ASGI (Asynchronous Server Gateway Interface) server that specifically serves ASGI applications, such as those built with FastAPI.
# It is responsible for handling the asynchronous aspects of the application, making it efficient for high-concurrency scenarios.

# Gunicorn is a WSGI (Web Server Gateway Interface) server. While it is not designed for handling asynchronous tasks directly, it can be used to serve synchronous WSGI applications, including FastAPI applications.
# Gunicorn is a pre-fork worker model server, meaning it spawns multiple worker processes to handle incoming requests concurrently. Each worker runs in a separate process and can handle one request at a time.