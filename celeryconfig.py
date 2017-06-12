from celery_dill_serializer import register_dill

register_dill()

task_serializer = 'dill'
result_serializer = 'dill'
accept_content = ['dill']

result_backend = 'redis://'

task_routes = {
    'persist_results': {'queue': 'db_write'},
    'persist_results_novelty': {'queue': 'db_write'},
    'check_if_done': {'queue': 'db_read'},
    'check_if_done_novelty': {'queue': 'db_read'},
    'reproduction_io': {'queue': 'db_read'},
    'reproduction_io_novelty': {'queue': 'db_read'},
}

result_expires = 15 * 60  # seconds

beat_schedule = {
    'celery.backend_cleanup': {
        'task': 'celery.backend_cleanup',
        'schedule': result_expires // 2,
        'args': (),
    },
}
