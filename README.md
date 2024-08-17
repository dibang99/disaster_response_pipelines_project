# Disaster Response Pipeline Project

## Project Overview
This project is part of the Udacity Data Scientist Nanodegree. The goal of this project is to build a Natural Language Processing (NLP) model to categorize messages sent during disasters. The model is trained on a dataset containing real messages that were sent during disaster events. The model is then used to classify new messages.

## Link Github
https://github.com/dibang99/disaster_response_pipelines_project

## Project Structure
Root Directory
- app/
    - template/
        - go.html
        - master.html
    - run.py # Main script to run the application
- data/
    - process.py # Script for data processing and cleaning
- image/
    - (Optional) Directory for image files or resources
- models/
    - train_classifier.py
- .DS_Store # macOS system file for folder attributes
- README.md # This README file
- requirements.txt # File listing the Python packages required


## Installation
```bash
   git clone https://github.com/dibang99/disaster_response_pipelines_project
```

```bash
pip install -r requirements.txt
```

## ETL
```python
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

Result:
```
Loading data...
    MESSAGES: data/disaster_messages.csv
    CATEGORIES: data/disaster_categories.csv
Cleaning data...
1.0    19906
0.0     6122
Name: related, dtype: int64
0.0    21554
1.0     4474
...
Name: other_weather, dtype: int64
0.0    20953
1.0     5075
Name: direct_report, dtype: int64
Saving data...
    DATABASE: data/DisasterResponse.db
Cleaned data saved to database!
```

## ML Pipeline
```python
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

Result:
```
Loading data...
    DATABASE: data/DisasterResponse.db
Building model...
Pipeline parameters:
 {'memory': ...}
Training model...
Evaluating model...
Category: related
Accuracy: 0.8000
F1 score: 0.7111
Precision: 0.6400
Recall: 0.8000

Category: request
             precision    recall  f1-score   support

        0.0       0.73      0.80      0.76        10
        1.0       0.78      0.70      0.74        10

avg / total       0.75      0.75      0.75        20


Accuracy: 0.9500
F1 score: 0.9256
Precision: 0.9025
Recall: 0.9500

Category: direct_report
             precision    recall  f1-score   support

        0.0       0.62      0.89      0.73         9
        1.0       0.86      0.55      0.67        11

avg / total       0.75      0.70      0.69        20

Accuracy: 0.7000
F1 score: 0.6939
Precision: 0.7484
Recall: 0.7000

Saving model...
    MODEL: models/classifier.pkl
Model saved successfully to models/classifier.pkl.
Trained model saved!
```

## Flask Web App
```python
python app/run.py
```

Open http://0.0.0.0:3001/ in your browser.

## Screenshots
### Homepage
![homepage](image/homepage.png)

### Visualizations
![visulize](image/visualize.png)

### Classify Message
![query](images/query_message.png)
