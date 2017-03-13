from celery_dill_serializer import register_dill

register_dill()

task_serializer = 'dill'
result_serializer = 'dill'
accept_content = ['dill']

result_backend = 'db+sqlite:///results.sqlite'
result_persistent = False

task_routes = {
    'persist_results': {'queue': 'db_io_tasks'},
    'check_if_done': {'queue': 'db_io_tasks'},
    'reproduction_io': {'queue': 'db_io_tasks'},
}

result_expires = 60 * 60  # seconds

beat_schedule = {
    'celery.backend_cleanup': {
        'tasks': 'celery.backend_cleanup',
        'schedule': 2 * result_expires,
        'args': (),
    },
}
