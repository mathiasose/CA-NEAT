import celery

app = celery.Celery('celery_app', config_source='celeryconfig', include=(
    'run_experiment',
    'run_neat',
    'ca',
    'problems',
    'geometry',
    'utils',
    'selection',
    'stagnation',
    'config',
    'patterns',
    'report',
))
