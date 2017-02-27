from celery_dill_serializer import register_dill

register_dill()

task_serializer = 'dill'
result_serializer = 'dill'
accept_content = ['dill']

result_backend = 'db+sqlite:///results.sqlite'
result_persistent = False

task_routes = {
    'finalize_generation': {'queue': 'db_io_tasks'},
}

# task_always_eager = True

# broker_url = 'sqla+sqlite:///celerydb.sqlite'
