import celery

from add_dill import add_dill

add_dill()

app = celery.Celery('celery_app', config_source='celeryconfig', include=(
    'run_experiment',
    'ca',
    'problems',
    'geometry',
    'utils',
    'selection',
))
