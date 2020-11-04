import requests

# Make a get request to get the latest position of the international space station from the opennotify api.
response = requests.get("https://cat-fact.herokuapp.com/facts")
#print(response.json(), '\n')

response = requests.get('https://jobs.github.com/positions.json?description=data+science&location=sf&page=1')

resp_json = response.json()
print(type(resp_json), len(resp_json))

response = requests.get('https://jobs.github.com/positions/c307e4ca-d6a6-11e8-8f6e-f00ef74f7cb0.json')
resp_json = response.json()

for job in resp_json:
    print(type(job))
    print('Job:', job, '\n')


