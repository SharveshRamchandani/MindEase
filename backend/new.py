import requests
import json 

# Define the URL and payload for the POST request
url = 'http://localhost:5000/predict'
payload = {'message': 'Hello, how are you?'}

# Convert the payload to JSON
json_payload = json.dumps(payload)

# Set the headers for the request
headers = {'Content-Type': 'application/json'}

# Send the POST request
response = requests.post(url, data=json_payload, headers=headers)

# Print the response
print(response.json())