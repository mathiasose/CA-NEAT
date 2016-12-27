from dill import dill
from kombu.serialization import pickle_protocol, pickle_loads, registry
from kombu.utils.encoding import str_to_bytes


def register_dill():
    def encode(obj, dumper=dill.dumps):
        return dumper(obj, protocol=pickle_protocol)

    def decode(s):
        return pickle_loads(str_to_bytes(s), load=dill.load)

    registry.register(
        name='dill',
        encoder=encode,
        decoder=decode,
        content_type='application/x-python-serialize',
        content_encoding='binary'
    )


register_dill()

CELERY_TASK_SERIALIZER = 'dill'
CELERY_ACCEPT_CONTENT = ['dill']

CELERY_RESULT_BACKEND = 'db+sqlite:///results.sqlite'
CELERY_RESULT_PERSISTENT = False

# CELERY_ALWAYS_EAGER = True

# BROKER_URL = 'sqla+sqlite:///celerydb.sqlite'
