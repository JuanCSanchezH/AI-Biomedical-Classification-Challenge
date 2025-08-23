install-local:
	pipenv install --dev
	pipenv run pre-commit install

create-environment:
	pipenv install --dev
	pipenv shell