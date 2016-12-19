import os
from io import BytesIO

from celery.app import shared_task
from pushbullet import Pushbullet

from database import Individual
from run_experiment import get_db
from secrets import PUSHBULLET_API_KEY
from utils import PROJECT_ROOT
from visualization.plot_fitness import plot_fitnesses_over_generations

PB = Pushbullet(PUSHBULLET_API_KEY)


def push_image(stream, file_name):
    file_data = PB.upload_file(stream, file_name, file_type='image/png')

    push = PB.push_file(**file_data)

    return push


@shared_task(name='send_results_via_pushbullet')
def send_results_via_pushbullet(db_path: str, scenario_id: int):
    db = get_db(db_path)
    plt = plot_fitnesses_over_generations(db=db, scenario_id=scenario_id, action='return')

    with BytesIO() as stream:
        plt.savefig(stream, format='png')
        stream.seek(0)
        timestamp = os.path.basename(db_path).replace('.db', '')
        push_image(stream, '{}.png'.format(timestamp))


@shared_task(name='send_message_via_pushbullet')
def send_message_via_pushbullet(title: str, body: str):
    PB.push_note(title=title, body=body)


if __name__ == '__main__':
    RESULTS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'problems', 'results'))
    file = os.path.join(RESULTS_DIR, 'replicate_tricolor/', '2016-11-09 15:13:23.893397.db')
    db_path = 'sqlite:///{}'.format(file)
    db = get_db(db_path)

    optimals = db.get_individuals(scenario_id=1).filter(Individual.fitness == 1.0)
    one_optimal_solution = optimals.first()

    if one_optimal_solution:
        print(one_optimal_solution)
