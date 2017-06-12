from sqlalchemy.exc import OperationalError

from ca_neat.database import get_db, Individual

X = """ generate_border_find_innovations_2017-04-21T20:14:55.212484     | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 generate_border_use_innovations_2017-06-01T22:41:06.610026      | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 generate_border_with_coord_input_2017-05-25T14:11:14.401192     | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 generate_mosaic_use_innovations_2017-06-02T15:52:21.403335      | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 generate_norwegian_flag_find_innovations_2017-05-26T17:21:47.72 | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 generate_norwegian_flag_find_innovations_2017-05-26T17:47:51.03 | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 generate_norwegian_flag_find_innovations_2017-05-26T18:37:47.79 | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 generate_norwegian_flag_find_innovations_2017-05-26T18:38:44.84 | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 generate_norwegian_flag_use_innovations_2017-05-30T21:19:53.326 | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 generate_norwegian_flag_use_innovations_2017-05-30T21:22:37.652 | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 generate_norwegian_flag_with_coord_input_2017-03-09T16:37:14.48 | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 generate_norwegian_flag_with_coord_input_2017-05-25T15:20:52.64 | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 generate_swiss_use_innovations_2017-06-02T16:07:27.409350       | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 generate_tricolor_use_innovations_2017-06-01T18:15:38.611671    | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 majority_2017-03-19T18:55:08.922502                             | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 majority_2017-03-24T01:39:14.176426                             | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 majority_2017-03-24T15:48:10.659623                             | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 majority_2017-03-27T14:54:14.480804                             | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 majority_2017-04-09T02:55:06.519066                             | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 majority_2017-04-09T03:00:41.882869                             | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 majority_2017-04-09T21:23:37.986163                             | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 majority_2017-04-17T12:56:25.176657                             | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 majority_2017-05-02T17:41:05.798487                             | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 majority_2017-05-02T17:42:29.143748                             | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 majority_2017-05-30T17:36:27.261999                             | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 postgres                                                        | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 replicate_norwegian_flag_use_innovations_2017-06-01T18:26:49.93 | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 replicate_norwegian_flag_use_innovations_2017-06-01T18:27:35.82 | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 replicate_swiss_use_innovations_2017-06-02T01:30:51.882720      | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 replicate_tricolor_use_innovations_2017-06-01T21:12:41.692797   | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 replicate_tricolor_use_innovations_2017-06-01T21:17:32.640588   | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 replicate_tricolor_use_innovations_2017-06-01T23:40:40.222102   | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 square_2017-05-20T09:01:11.261596                               | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 square_2017-05-20T11:54:24.706483                               | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 swiss_different_settings_2017-05-21T19:37:44.047610             | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 swiss_different_settings_2017-05-24T15:50:57.508660             | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 swiss_different_settings_2017-05-24T15:53:30.405115             | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 synchronization_2017-04-08T21:52:08.875799                      | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 synchronization_2017-05-05T18:21:54.491165                      | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 synchronization_2017-05-06T02:18:24.967136                      | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
 synchronization_2017-05-06T17:00:23.012278                      | ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 | """

for line in X.split('\n'):
    line = line.strip().rstrip('| ose      | UTF8     | en_US.UTF-8 | en_US.UTF-8 |').strip()

    if 'use_innovations' not in line:
        continue

    try:

        DB_PATH = f'postgresql+psycopg2:///{line}'

        db = get_db(DB_PATH, create_if_nonexistent=False)
    except OperationalError:
        continue

    print(line)

    for scenario in db.get_scenarios():
        print(f'Scenario {scenario.id}:',
              db.get_individuals(scenario_id=scenario.id).filter(Individual.fitness == 1.0).count())

    print()
