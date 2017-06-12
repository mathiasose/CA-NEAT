import os
from io import BytesIO

from celery.app import shared_task
from pushbullet import Pushbullet

from ca_neat.database import get_db

try:
    from secrets import PUSHBULLET_API_KEY
except ImportError:
    PUSHBULLET_API_KEY = None

if PUSHBULLET_API_KEY:
    PB = Pushbullet(PUSHBULLET_API_KEY)
else:
    PB = None

AUTO_RETRY = {
    'autoretry_for': (Exception,),
    'retry_kwargs': {'countdown': 30},
}


def push_image(stream, file_name):
    if PB is None:
        return

    file_data = PB.upload_file(stream, file_name, file_type='image/png')

    push = PB.push_file(**file_data)

    return push


@shared_task(name='send_results_via_pushbullet', **AUTO_RETRY)
def send_results_via_pushbullet(db_path: str, scenario_id: int):
    if PB is None:
        return

    db = get_db(db_path)
    plt = plot_fitnesses_over_generations(db=db, scenario_id=scenario_id, action='return')

    with BytesIO() as stream:
        plt.savefig(stream, format='png')
        stream.seek(0)
        timestamp = os.path.basename(db_path).replace('.DB', '')
        push_image(stream, '{}.png'.format(timestamp))


@shared_task(name='send_message_via_pushbullet', **AUTO_RETRY)
def send_message_via_pushbullet(title: str, body: str):
    if PB is None:
        return

    PB.push_note(title=title, body=body)
