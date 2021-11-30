import json

# a Python object (dict):
x = {'Angry': 0.049574226, 'Fear': 0.046781816, 'Happy': 0.017478904, 'Neutral': 0.76315033, 'Sad': 0.119851045, 'Surprised': 0.0031637386}

# convert into JSON:
y = json.dumps(str(x))

# the result is a JSON string:
print(y)