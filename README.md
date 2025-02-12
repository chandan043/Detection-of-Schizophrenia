# Schizophrenia Detector API

A FastAPI-based backend API to detect the likelihood of schizophrenia based on input comments. The API serves a prediction model and includes an optional frontend for user interaction.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [License](#license)

---

## Features
- **Prediction API**: Analyze text data to assess schizophrenia likelihood using a trained model.
- **Frontend Support**: Serves an HTML file for interaction (optional).
- **CORS Middleware**: Allows secure cross-origin requests.
- **Error Handling**: Handles missing input and system errors gracefully.

---

## Requirements
- Python 3.8 or later
- FastAPI
- Pydantic
- Uvicorn
- Your custom prediction model file (`model.py`)

Install dependencies:
```bash
pip install fastapi uvicorn pydantic
```

## Installation
Clone the Repository:

```bash
git clone https://github.com/chandan043/schizophrenia-detector-api.git
cd schizophrenia-detector-api
```
Install the dependencies:

```bash
pip install -r requirements.txt
```
Ensure the following files are in place:

- model.py: Contains the predict_comments function for prediction.
- frontend.html: (Optional) A simple HTML file to serve as the frontend.
- Run the application:

```bash
uvicorn main:app --reload
```
# Usage
Starting the API
Run the following command to start the server:

```bash
uvicorn main:app --reload
```
The API will be available at http://127.0.0.1:8000.

Accessing the Frontend
Open http://127.0.0.1:8000/ in your browser to view the optional frontend.
## API Endpoints

### `GET /`
- **Description**: Serves the optional `frontend.html` file.
- **Response**: Returns the contents of the HTML file if it exists.
- **Error**: 
  - `404 Not Found`: If the `frontend.html` file is missing.

---

### `POST /predict`
- **Description**: Predicts the likelihood of schizophrenia based on user-provided comments.
- **Request Body**:
    ```json
    {
        "comments": ["sample comment 1", "sample comment 2"]
    }
    ```
- **Response**:
    ```json
    {
        "predictions": [0.85, 0.45]
    }
    ```
    - The `predictions` array contains the likelihood scores for each comment.
- **Error Responses**:
  - `400 Bad Request`: If the input list is empty.
    - Example response:
      ```json
      {
          "detail": "Input list cannot be empty"
      }
      ```
  - `500 Internal Server Error`: For unexpected issues.
    - Example response:
      ```json
      {
          "detail": "An internal error occurred"
      }
      ```
# License
This project is licensed under the MIT License. See the LICENSE file for details.

