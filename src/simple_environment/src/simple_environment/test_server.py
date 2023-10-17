"""Script for testing search/server.py.

Usage:
    # Open three terminals. In the first terminal, run:
    roscore

    # In another terminal, run:
    python search/server.py -c search/config/experiment/experiment.tml

    # In a third terminal, run:
    python test_server.py
"""
import requests
import json

# See here for more info:
# https://realpython.com/python-requests/
try:
    response = requests.post(
       "http://localhost:5000/evaluate",
        data={
            # Post requests are not designed to take in nested data, so we have to
            # package the data on our own with JSON.
            "scenario_params": json.dumps([
                {
                    "function": "scenario_generate",
                    "kwargs": {
                        "p1_x": 0.05,
                        "p1_y": 0.05,
                        "p2_x": 0.15,
                        "p2_y": 0.15,
                        "disturbances": [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                    },
                },
            ] * 2)
        },
    )

    # If the response was successful, no Exception will be raised.
    response.raise_for_status()
except requests.exceptions.HTTPError as http_err:
    print('HTTP error occurred ' + str(http_err))
    print(response.content)
except Exception as err:
    print('Other error occurred ' + str(err))
else:
    print('Success!')
    print(response.json())
