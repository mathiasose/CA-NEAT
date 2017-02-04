import celery

app = celery.Celery('celery_app', config_source='celeryconfig', include=(
    'ca_neat.ca',
    'ca_neat.ga',
    'ca_neat.geometry',
    'ca_neat.patterns',
    'ca_neat.problems',
    'ca_neat.run_experiment',
    'ca_neat.utils',
    'ca_neat.config',
    'ca_neat.report',
))
