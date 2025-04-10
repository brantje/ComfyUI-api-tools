# ComfyUI API Tools

Add some extra endpoints to ComfyUI to manage / monitor remotely.


## Get started
1. go to custom_nodes
   ```bash
    cd custom_nodes
   ```
2. clone the repository
   ```bash
    git clone https://github.com/brantje/ComfyUI-api-tools
    cd ComfyUI-api-tools
   ```

## API Endpoints

| Endpoint                                 | Method | Description                                            | Parameters |
|------------------------------------------|--------|--------------------------------------------------------|------------|
| `/api-tools/v1/models`                   | GET | List all model folders                                 | None |
| `/api-tools/v1/models/{folder}`          | GET | Get all models in the folder                           | None |
| `/api-tools/v1/models/{folder}/refresh`  | POST | Refresh the list of models and return the updated list | None |
| `/api-tools/v1/models/{folder}/{model}`  | DELETE | Remove a model from the file system                    | None |
| `/api-tools/v1/images/output`            | GET | List all the output images                             | `temp`: (boolean) When `true`, only lists temporary output images generated by the `PreviewImage` node |
| `/api-tools/v1/images/output/{filename}` | DELETE | Delete the output image with the given filename        | `temp`: (boolean) When `true`, only deletes temporary output images generated by the `PreviewImage` node |
| `/api-tools/v1/images/input`             | GET | List all the input images                              |
| `/api-tools/v1/images/input/{filename}`  | DELETE | Delete the input image with the given filename         | None |
| `/api-tools/v1/metrics`                  | GET | Prometheus metrics                                     | None |

