

def add_dill():
    import dill
    import kombu
    from kombu.five import BytesIO
    registry = kombu.serialization.registry
    kombu.serialization.pickle = dill

    registry.unregister('pickle')

    def pickle_loads(s, load=dill.load):
        # used to support buffer objects
        return load(BytesIO(s))

    def pickle_dumps(obj, dumper=dill.dumps):
        return dumper(obj, protocol=kombu.serialization.pickle_protocol)

    registry.register('pickle', pickle_dumps, pickle_loads,
                      content_type='application/x-python-serialize',
                      content_encoding='binary')

    # there are a couple of other places that celery uses pickle *without* referencing the registry, and imports
    # the module on init in a way that doesn't update if we change it in kombu.serialization; so we monkey patch
    # each one

    import celery.worker
    celery.worker.state.pickle = dill

    import celery.concurrency.asynpool
    celery.concurrency.asynpool._pickle = dill
