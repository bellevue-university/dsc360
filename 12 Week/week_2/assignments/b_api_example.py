import requests

# Make a get request to get the latest position of the international space station from the opennotify api.
response = requests.get("http://api.open-notify.org/iss-now.json")
# Print the status code of the response.
print(response.status_code)

# Ask for something that doesn't exist
response = requests.get("http://api.open-notify.org/iss-pass")
print(response.status_code)

# Right endpoint, bad request
response = requests.get("http://api.open-notify.org/iss-pass.json")
print(response.status_code)

# Set up the parameters we want to pass to the API.
# This is the latitude and longitude of New York City.
parameters = {"lat": 40.71, "lon": -74}
# Make a get request with the parameters.
response = requests.get("http://api.open-notify.org/iss-pass.json", params=parameters)
# Print the content of the response (the data the server returned)
print(response.content)
# This gets the same data as the command aboveresponse = requests.get("http://api.open-notify.org/iss-pass.json?lat=40.71&lon=-74")
print(response.content, '\n')

# Decode the response, no real change to the output, but it's decoded now
response.content.decode("utf-8")
print(response.content, '\n')

# JSON Detour
# Make a list of fast food chains.
best_food_chains = ["Taco Bell", "Shake Shack", "Chipotle"]
# This is a list.
print(type(best_food_chains))
# Import the json library
import json
# Use json.dumps to convert best_food_chains to a string.
best_food_chains_string = json.dumps(best_food_chains)
# We've successfully converted our list to a string.
print(type(best_food_chains_string))
# Convert best_food_chains_string back into a list
print(type(json.loads(best_food_chains_string)))
# Make a dictionary
fast_food_franchise = {
"Subway": 24722,
"McDonalds": 14098,
"Starbucks": 10821,
"Pizza Hut": 7600
}
# We can also dump a dictionary to a string and load it.
fast_food_franchise_string = json.dumps(fast_food_franchise)
print(type(fast_food_franchise_string), '\n')

# Back to the API
# Make the same request we did earlier, but with the coordinates of San Francisco instead.
parameters = {"lat": 37.78, "lon": -122.41}
response = requests.get("http://api.open-notify.org/iss-pass.json", params=parameters)
# Get the response data as a python object. Verify that it's a dictionary.
data = response.json()
print(type(data))
print(data, '\n')

# Headers is a dictionary
print(response.headers)
# Get the content-type from the dictionary.
print(response.headers["content-type"], '\n')

# One more example
# Get the response from the API endpoint.
response = requests.get("http://api.open-notify.org/astros.json")
data = response.json()
# 9 people are currently in space.
print(data["number"])
print(data)