
import requests


url = "https://rajaaliakhtar108--evo-service-evoserviceapi.modal.run/score_delta"

data = {
    "ref_sequence": "ATCGATCG",
    "alt_sequence": "ATCGATCA"
}

response = requests.post(url, json=data)

if response.ok:
    print(response.json())
else:
    print("Error:", response.status_code, response.text)



# 1. URL for the batch scoring endpoint
url = "https://rajaaliakhtar108--evo-service-evoserviceapi.modal.run/score_batch"

# 2. Prepare your data payload (list of sequence pairs)
data = {
    "pairs": [
        {"ref_sequence": "ATCGATCG", "alt_sequence": "ATCGATCA"},
        {"ref_sequence": "GCTAGCTA", "alt_sequence": "GCTAGCTG"}
    ]
}

# 3. Send the POST request to the API
response = requests.post(url, json=data)

# 4. Check and print the response
if response.ok:
    results = response.json()
    print("Batch scoring results:")
    for i, result in enumerate(results):
        print(f"Pair {i+1}:")
        print(f"  ref_likelihood = {result['ref_likelihood']}")
        print(f"  alt_likelihood = {result['alt_likelihood']}")
        print(f"  delta_score    = {result['delta_score']}")
else:
    print("Error:", response.status_code, response.text)

import requests

url = "https://rajaaliakhtar108--evo-service-evoservice-api.modal.run/score_delta"
data = {
    "ref_sequence": "ATCGATCG",
    "alt_sequence": "ATCGATCA"
}

r = requests.post(url, json=data)
print(r.status_code, r.text)
