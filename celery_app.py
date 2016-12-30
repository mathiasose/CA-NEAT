import celery

app = celery.Celery('celery_app', config_source='celeryconfig', include=(
    'ca',
    'ga',
    'geometry',
    'patterns',
    'problems',
    'run_experiment',
    'utils',
    'config',
    'report',
))
