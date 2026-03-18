CREATE DATABASE apache_db;
CREATE DATABASE mlops_db;

\c mlops_db;

CREATE TABLE train_data (
    age FLOAT, sex INTEGER, cp INTEGER, trestbps FLOAT, chol FLOAT,
    fbs INTEGER, restecg INTEGER, thalach FLOAT, exang INTEGER,
    oldpeak FLOAT, slope FLOAT, ca INTEGER, thal VARCHAR, target INTEGER
);

CREATE TABLE test_data (
    age FLOAT, sex INTEGER, cp INTEGER, trestbps FLOAT, chol FLOAT,
    fbs INTEGER, restecg INTEGER, thalach FLOAT, exang INTEGER,
    oldpeak FLOAT, slope FLOAT, ca INTEGER, thal VARCHAR, target INTEGER
);

CREATE TABLE api_logs (
    id SERIAL PRIMARY KEY,
    request_payload JSONB,
    prediction FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
