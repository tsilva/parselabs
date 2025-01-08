# labs-parser

[Logo](logo.jpg)

## Installation

To set up the conda environment using the `environment.yml` file, run the following command:

```sh
conda env create -f environment.yml
```

This will create a new conda environment with all the dependencies specified in the `environment.yml` file.

## Updating the Environment

If you need to update the conda environment with any changes made to the `environment.yml` file, run:

```sh
conda env update --file environment.yml --prune
```

The `--prune` flag will remove any dependencies that are no longer required.

## Testing

```sh
python -m tests.test_pipeline
```

## TODO

- [ ] Prescan for missing enum values using function call
- [ ] Fix expected.csv
- [ ] Run multiple parsers in parallel and merge their results
- [ ] Add unit test for single document parsing
- [ ] Upgrade to use langchain (abstract underlying model)