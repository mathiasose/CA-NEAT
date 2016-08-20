import celery

from add_dill import add_dill

add_dill()

app = celery.Celery('celery_app', include=['run_experiment'])
