# BoschTrainingDataScience

## DVC Workflow (Data + Code Versioning Together)

1. Initialize dependencies and DVC:

	/bin/python3 -m pip install --user dvc
	~/.local/bin/dvc init

2. Track dataset with DVC:

	git rm --cached automobile_dataset.csv
	~/.local/bin/dvc add automobile_dataset.csv

3. Reproduce full ML pipeline:

	~/.local/bin/dvc repro

4. Optional: configure remote storage and push tracked data artifacts:

	~/.local/bin/dvc remote add -d localstorage /tmp/bosch-dvc-storage
	~/.local/bin/dvc push

## Random Forest Workflow

1. Train the model and generate the pickle pipeline:

	/bin/python3 rf_feature_engineering.py

2. Execute the pickle file for sample inputs and recommendations:

	/bin/python3 run_pickle_recommendations.py

## DVC Pipeline Artifacts

- Data pointer: automobile_dataset.csv.dvc
- Pipeline definition: dvc.yaml
- Pipeline lock: dvc.lock
- Training metrics: metrics.json (DVC metrics)
- Model artifact: random_forest_pipeline.pkl (DVC output)
- Inference output: recommendations.csv (DVC output)

## FastAPI Model Serving

1. Start the API server:

	/bin/python3 -m uvicorn serve_model_api:app --host 0.0.0.0 --port 8000

2. Open interactive API docs (Swagger UI):

	http://127.0.0.1:8000/docs

3. Open raw OpenAPI schema:

	http://127.0.0.1:8000/openapi.json

4. Generate a static Swagger document file:

	/bin/python3 generate_swagger.py

Example prediction request:

curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
	 "Make": "Toyota",
	 "Model": "Camry",
	 "Year": 2021,
	 "Fuel_Type": "Petrol",
	 "Transmission": "Automatic",
	 "Engine_Size": 2.0,
	 "Mileage": 42000,
	 "Horsepower": 165,
	 "Torque": 155,
	 "Owners": 1,
	 "Accident_History": 0,
	 "Service_History": "Full Service",
	 "Color": "White",
	 "Body_Type": "Sedan",
	 "Drivetrain": "FWD",
	 "Fuel_Efficiency": 36,
	 "Location": "CA",
	 "Listed_Price": 18500
  }'

## Docker Deployment

1. Build Docker image:

	docker build -t bosch-car-price-api:latest .

2. Run container:

	docker run -d --name bosch-car-price-api-container -p 8000:8000 bosch-car-price-api:latest

3. Check health endpoint:

	curl http://127.0.0.1:8000/health

4. Access Swagger UI:

	http://127.0.0.1:8000/docs

5. Stop and remove container when done:

	docker rm -f bosch-car-price-api-container