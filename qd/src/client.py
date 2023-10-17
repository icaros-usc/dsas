"""Provides the Client for interacting with the server."""
import json
import logging
import urllib.parse
from dataclasses import asdict

import numpy as np
import requests

from src.scenario.scenario_config import ScenarioConfig
from src.scenario.scenario_result import ScenarioResult

logger = logging.getLogger(__name__)


class Client:
    """The Client provides an interface for querying the server for
    evaluations."""

    def __init__(self, address):
        if address.endswith("/"):
            address = address[:-1]
        self.address = address

    def _log_address(self, path):
        address = urllib.parse.urljoin(self.address, path)
        logger.info("Querying %s", address)
        return address

    def hello(self):
        response = requests.get(self._log_address("/"))
        return response.text

    def ncores(self):
        """Number of cores used by the server."""
        response = requests.get(self._log_address("/ncores"))
        return int(response.text)

    def log(self, message):
        requests.post(
            self._log_address("/log"),
            data={"message": message},
        )

    def evaluate(self, solution_batch, config: ScenarioConfig):
        """Evaluates a batch of solutions.

        See here for more info on requests: https://realpython.com/python-
        requests/
        """
        logger.info(
            "Querying server for evaluation of %d solutions with "
            "%d evaluation(s) per solution",
            len(solution_batch),
            config.n_evals,
        )

        # Achieve multiple evals by sending the solutions multiple times.
        aug_solution_batch = np.repeat(solution_batch,
                                       repeats=config.n_evals,
                                       axis=0)

        request_data = {
            # Post requests are not designed to take in nested data, so we
            # have to package the data on our own with JSON.
            "scenario_params":
                json.dumps([{
                    "function": config.scenario_function,
                    "features": config.measure_names,
                    "solution": s,
                    "kwargs": asdict(config.kwargs_to_pass),
                } for i, s in enumerate(aug_solution_batch.tolist())])
        }
        logger.debug("Request data:\n%s", request_data)

        address = self._log_address("/evaluate")

        try:
            response = requests.post(address, data=request_data)

            # If the response was successful, no Exception will be raised.
            response.raise_for_status()
        except Exception as e:  # pylint: disable = broad-except
            logger.info(
                "Error occurred and will be re-raised. Response content:\n%s",
                response.content)
            raise e
        else:
            logger.info("Successfully completed query")

        logger.info("Processing evaluations")

        data = response.json()
        logger.debug("Response data:\n%s", data)

        results = []
        for i, solution in enumerate(solution_batch):
            data_chunk = data[i * config.n_evals:(i + 1) * config.n_evals]

            # Check for failure.
            if any(d["status"] == "error" for d in data_chunk):
                r = ScenarioResult(solution=solution)
                r.failed = True
                r.error_message = next(
                    d for d in data_chunk if d["status"] == "error").get(
                        "message", "No error message available")
                results.append(r)
                logger.info("Solution %d failed with message: %s", i,
                            r.error_message)
                continue

            # Otherwise, assemble the result.
            results.append(
                ScenarioResult.from_raw(
                    solution=solution,
                    objs=np.array([d["fitness"] for d in data_chunk]),
                    measures=np.array([d["features"] for d in data_chunk]),
                    metadata=[{
                        "goals": d.get("points", None),
                        "obstacles": d.get("obstacles", None),
                        "human_trajectory": d.get("human_trajectory", None),
                        "robot_trajectory": d.get("robot_trajectory", None),
                    } for d in data],
                    opts={"aggregation": config.aggregation_type},
                ))

        return results


class MockClient(Client):
    """Mock version of the Client for testing."""

    def hello(self):
        return "Test Hello World"

    def ncores(self):
        return 4

    def log(self, message):
        pass

    def evaluate(self, solution_batch, config: ScenarioConfig):
        return [
            ScenarioResult.from_raw(
                solution=solution,
                objs=np.zeros(config.n_evals),
                measures=np.zeros((config.n_evals, len(config.measure_names))),
                metadata=[None for _ in range(config.n_evals)],
                opts={"aggregation": config.aggregation_type},
            ) for solution in solution_batch
        ]
