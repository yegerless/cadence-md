# Project Setup Instruction

Follow the instructions below to setup the project.

### Installation Steps

#### 1. Clone the Repository

Clone the project repository from GitHub to your local machine:

```bash
git clone git@github.com:CORESIGHT-Health/simple-randomiser.git
cd simple-randomiser
```

#### 2. Init virtual environment and install dependencies

Run next command:

```
poetry install
```

If you want to install all dependencies (include development) use next command:

```
poetry install --with dev
```

#### 3. Install pre-commit hook

Pre-commit tool configured yet, you need only install git hook:

```bash
poetry run pre-commit install
```

#### 4. Get data from DVC remote storage

In that project we use Yandex Cloud object storage as dvc remote storage to
store texts corpus of russian clinical recommendations. The basic settings for
connecting to the remote storage of the DVC are already written in the file
.dvc/config. Use next commands to set up DVC remote storage security settings
and download texts corpus to your local computer.

```bash
# Add credentials for connection to remote DVC storage or use .dvc/config.local.example
poetry run dvc remote modify --local yandex access_key_id '<your_access_key_id>'
poetry run dvc remote modify --local yandex secret_access_key '<your_secret_access_key>'

# Download data from DVC remote storage
poetry run dvc pull
```

#### Additional info

Always run pre-commit before create new commits:

```bash
poetry run pre-commit run -av
```

All pre-commit checks should not be red.

Export dependencies from poetry in requirements.txt for running project in
docker:

```bash
poetry export --without-hashes --format=requirements.txt --without dev > requirements.txt
```
