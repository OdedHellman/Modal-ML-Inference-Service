# Flux ML Inference Service

![Flux Inference Service](https://eikona-io-ml.s3.eu-north-1.amazonaws.com/images/flux-1756310122703.jpg)

## Overview

This project provides a high-performance, and scalable project for serving ML models. It is built on the [Modal](https://modal.com/) serverless platform and utilizes the powerful `black-forest-labs/FLUX.1-schnell` model for generating images from text prompts. The service is exposed via a FastAPI endpoint and is designed for parallel processing and on-demand scalability.

## Features

- **Scalable & Serverless:** Built on Modal for automatic scaling based on demand, so you only pay for what you use.
- **Parallel Processing:** Can handle multiple inference requests concurrently using a pool of GPU workers.
- **Flexible & Configurable:** The API allows for a wide range of sampling parameters, including different samplers, step counts, and classifier-free guidance (CFG) scales.
- **Cloud Integration:** Automatically saves generated images to an AWS S3 bucket and returns public URLs.
- **Asynchronous by Design:** Built with `asyncio` and FastAPI for efficient handling of concurrent requests.


## Files
-   **`app.ipynb`**: A Jupyter notebook for experimentation and demonstration (use this for testing).
-   **`flux_serve.py`**: The main application file that defines the Modal app, the FastAPI endpoint, and the `FluxModel` class for inference.
-   **`schemas/schemas.py`**: Defines the Pydantic models for the API request and response.
-   **`utils/`**: Contains helper functions for model optimization, sampling, and other utilities.



## Installation & Deploy

-  Follow `app.ipynb` instructions.


## API Reference

### `/generate`

**Method:** `POST`

**Description:** Generates images from a text prompt.

**Request Body:**

The request body should be a JSON object with the following fields:

-   `prompt` (string, required): The text prompt to generate images from.
-   `nof_images` (integer, optional, default: 16): The number of images to produce.
-   `gpus` (integer, optional, default: 1): The number of GPUs to use for parallel processing.
-   `diversity` (object, optional): A dictionary to control the diversity of the generated images.
    -   `samplers` (array of strings, optional): A list of samplers to use.
    -   `nof_steps` (array of integers, optional): A list of step counts for sampling.
    -   `cfg_scales` (array of floats, optional): A list of classifier-free guidance scales.
    -   `seeds` (string, optional): How to seed the runs. Can be "fixed", "random", or "mix".
-   `stream` (boolean, optional, default: False): If true, stream links as they’re ready.
-   `top_k` (integer, optional): If set, return only the best K images after scoring.

**Response:**

A JSON object with the following fields:

-   `prompt` (string): The prompt that was used to generate the images.
-   `count` (integer): The number of generated images.
-   `images` (array of objects): A list of generated images, each with a URL and sampling parameters.
-   `latency_ms` (integer): The total time taken to generate the images in milliseconds.

## Dependencies

-   [modal](https://modal.com/)
-   [fastapi](https://fastapi.tiangolo.com/)
-   [torch](https://pytorch.org/)
-   [diffusers](https://huggingface.co/docs/diffusers/index)
-   [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
