import os

from sqlalchemy.sql import exists
from sqlalchemy.sql.functions import func

from ca_neat.database import Individual, get_db
from ca_neat.utils import PROJECT_ROOT

N = 100

if __name__ == '__main__':

    db_path = DB_PATH = 'postgresql+psycopg2:///generate_norwegian_flag_with_coord_input_2017-03-09T16:37:14.48'
    db = get_db(db_path)

    session = db.Session()

    scenarios = db.get_scenarios(session=session)

    for scenario in scenarios:
        try:
            x = session.query(
                exists()
                    .where(Individual.scenario_id == scenario.id)
                    .where(Individual.fitness >= 1.0)
            ).scalar()
            y = session.query(func.max(Individual.generation)).filter_by(scenario_id=scenario.id).scalar()

            if y is None:
                print(scenario.id, 'skip')
                continue

            yy = (y // N) * N - 15
            print(scenario.id, x, y, yy)

            q = session.query(Individual) \
                .filter(Individual.scenario_id == scenario.id) \
                .filter(Individual.generation < yy) \
                .filter(Individual.genotype != None) \
                .filter((Individual.generation % N) != 0)

            assert q.filter(Individual.generation == y).count() == 0

            print(q.count())

            q.update(values={
                'genotype': None,
            })

            session.commit()
        except KeyboardInterrupt:
            break

    session.close()
