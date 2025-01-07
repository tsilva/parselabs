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

## TODO

- [ ] Instead of adding hash as file prefix, create an hashes.json file at the root mapping the hash to the file name
- [ ] Add unit test for single document parsing