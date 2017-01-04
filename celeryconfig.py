from dill import dill
from kombu.serialization import pickle_loads, pickle_protocol, registry
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

task_serializer = 'dill'
event_serializer = 'dill'
result_serializer = 'dill'
accept_content = ['dill']

result_backend = 'db+sqlite:///results.sqlite'
result_persistent = False

# task_always_eager = True

# broker_url = 'sqla+sqlite:///celerydb.sqlite'
