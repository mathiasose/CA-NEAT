import os

from database import Individual, get_db
from run_experiment import initialize_generation
from run_neat import sort_into_species
from utils import PROJECT_ROOT, pluck

if __name__ == '__main__':
    problem_name, db_file = ('generate_swiss_flag', '2016-11-18 21:48:15.168123.db')
    from problems.generate_swiss_flag import CA_CONFIG, NEAT_CONFIG, FITNESS_F, PAIR_SELECTION_F

    db_path = 'sqlite:///{}'.format(os.path.join(PROJECT_ROOT, 'problems', 'results', problem_name, db_file))
    db = get_db(db_path)

    session = db.Session()

    for l in """1 149 True
2 400 False
3 400 False
4 400 False
5 79 True
6 44 True
7 66 True
8 400 False
9 69 True
10 38 True
11 38 True
12 25 True
13 17 True
14 400 False
15 400 False
16 8 True
17 400 False
18 165 True
19 43 True
20 255 True
21 102 True
22 8 True
23 45 True
24 26 True
25 400 False
26 400 False
27 400 False
28 398 True
29 40 True
30 400 False
31 294 True
32 400 False
33 400 False
34 1 True
35 400 False
36 371 True
37 136 True
38 216 True
39 400 False
40 400 False
41 400 False
42 15 True
43 51 True
44 240 True
45 400 False
46 125 True
47 152 True
48 59 True
49 19 True
50 2 True
51 400 False
52 11 True
53 400 False
54 400 False
55 400 False
56 400 False
57 62 True
58 400 False
59 84 True
60 44 True
61 149 True
62 151 True
63 400 False
64 294 True
65 400 False
66 277 True
67 400 False
68 135 True
69 400 False
70 119 True
71 59 True
72 11 True
73 9 True
74 113 True
75 72 True
76 7 True
77 107 True
78 194 True
79 400 False
80 229 True
81 400 False
82 400 False
83 85 True
84 24 True
85 334 True
86 161 True
87 400 False
88 64 True
89 15 True
90 30 True
91 79 True
92 5 True
93 99 True
94 400 False
95 15 True
96 9 True
97 25 True
98 35 True
99 114 True
100 215 True""".strip().split('\n'):
        l = l.strip()

        if 'True' in l:
            continue

        scenario_id, generation_n, _ = l.split(' ')
        scenario_id = int(scenario_id)
        generation_n = int(generation_n)

        if generation_n >= 500:
            continue

        print(scenario_id, generation_n)

        generation_n -= 1
        genotypes = list(pluck(db.get_generation(scenario_id, generation=generation_n), 'genotype'))

        assert genotypes

        to_purge = db.get_individuals(scenario_id).filter(Individual.generation >= generation_n)
        print(scenario_id, 'deleting', to_purge.count())
        to_purge.delete()

        session.commit()

        species = sort_into_species(genotypes)

        initialize_generation(
            db_path=db_path,
            scenario_id=scenario_id,
            generation=generation_n,
            genotypes=genotypes,
            fitness_f=FITNESS_F,
            pair_selection_f=PAIR_SELECTION_F,
            neat_config=NEAT_CONFIG,
            ca_config=CA_CONFIG,
        )
