# Flask Repository

This repository contains a Flask application, a nice interface for a climate change prediction tool which you can run locally on your computer.

To run this project on your laptop please follow the instructions provided below:
1. Install dependencies (git, python, pip) - find tutorial on internet if you don't know how to

2. Clone project to you PC.
```
git clone https://github.com/osanan25/Flask
```

3. Open project folder
```
cd Flask
```

4. Open terminal in the project directory and run command
```
python -m venv venv
```
or
```
python3 -m venv venv
```
depending on how you installed python

6. Activate venv with command
- windows: 
```
venv\Scripts\activate
```

- mac and linux:
```
source venv/bin/activate
```

7. Install pip-compile tool
```
pip install pip-tools
```

8. Create dependencies file
```
pip-compile --output-file=requirements.txt requirements.in
```

9. Install all dependencies with command:
```
pip3 install -r requirements.txt
```

Now you can run flask app from this terminal window by using command:
```
flask run
```
# Latest updates

21.06.2023
 - All models now lacated in a separate file
 - 2 new models were added: ARIMA and MLP regresser. There are mot suitable models for time series with small dataset.