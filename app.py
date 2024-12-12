import uvicorn
from fastapi import FastAPI
import joblib
from loan import loan

app = FastAPI()
joblib_in = open("loan-recommender.joblib","rb")
model=joblib.load(joblib_in)


@app.get('/')
def index():
    return {'message': 'Loan Recommender ML API'}

@app.post('/loan/predict')
def predict_loan_approval(data:loan):
    data = data.dict()
    age=data['age']
    gender=data['gender']

    prediction = model.predict([[age, gender]])
    
    return {
        'prediction': prediction[0]
    }
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)