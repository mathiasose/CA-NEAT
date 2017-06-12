from collections import defaultdict
from random import uniform, choice
from statistics import mean, mode

from neat.nn import create_feed_forward_phenotype
from tqdm._tqdm import tqdm

from ca_neat.ca.analysis import serialize_cppn_rule
from ca_neat.database import get_db, Individual
from ca_neat.ga.serialize import deserialize_gt
from ca_neat.problems.majority import create_binary_pattern
from ca_neat.problems.majority import CA_CONFIG, NEAT_CONFIG, fitness_f

if __name__ == '__main__':
    DB_PATH = 'postgresql+psycopg2:///majority_2017-05-02T17:42:29.143748'
    db = get_db(DB_PATH)

    session = db.Session()

    assert session.query(Individual).count()

    scenarios = list(db.get_scenarios(session))

    behaviors = defaultdict(list)
    for scenario in tqdm(scenarios):
        xs = db.get_individuals(scenario_id=scenario.id, session=session) \
            .filter(Individual.fitness == 1.0) \
            .order_by(Individual.generation)

        if not xs.count():
            continue

        for x in xs:
            gt = deserialize_gt(x.genotype, NEAT_CONFIG)

            cppn = create_feed_forward_phenotype(gt)

            k = ''.join(serialize_cppn_rule(cppn, CA_CONFIG)[1])

            if k in behaviors.keys():
                continue
            else:
                behaviors[k] = {
                    'cppn': cppn,
                    'f': gt.fitness,
                    'test_scores': []
                }

    print(session.query(Individual).filter(Individual.fitness==1.0).count())
    print(len(behaviors))

    K = 1000
    I = 25
    TESTS = {
        49: [],
        99: [],
        149: [],
    }

    for n in TESTS.keys():
        for _ in range(K):
            while True:
                r = uniform(0, 1)

                p = create_binary_pattern(alphabet=CA_CONFIG.alphabet, length=n, r=r)

                if p in TESTS[n]:
                    continue
                else:
                    TESTS[n].append((p, mode(p)))
                    break

    for i, k in enumerate(behaviors.keys()):
        for n in sorted(TESTS.keys()):
            CA_CONFIG.etc = {
                'test_patterns': TESTS[n],
                'I': I,
            }
            CA_CONFIG.iterations = 2 * n
            cppn = behaviors[k]['cppn']
            behaviors[k]['test_scores'].append(fitness_f(cppn, CA_CONFIG))
        behaviors[k]['mean'] = mean(behaviors[k]['test_scores'])
        print(i, behaviors[k])

    import matplotlib.pyplot as plt
    import seaborn
    from tqdm._tqdm import tqdm

    seaborn.set_style('whitegrid')

    fig = plt.figure()

    vs = sorted(behaviors.values(), key=lambda x: (x['f'], x['mean'], x['test_scores']))

    T = [x['f'] for x in vs]
    M = [x['mean'] for x in vs]

    A = [x['test_scores'][0] for x in vs]
    B = [x['test_scores'][1] for x in vs]
    C = [x['test_scores'][2] for x in vs]

    plt.ylim(min(min(T), min(A), min(B), min(C)), 1.0)

    print(A)
    print(B)
    print(C)

    r = list(range(1, len(T) + 1))
    # plt.plot(r, T, '.', label='Training set (n=49, k=100)')
    plt.plot(r, M, 'D', label='Mean of test sets')
    plt.plot(r, A, '*', label='Test set (n=49, k=1000)')
    plt.plot(r, B, '*', label='Test set (n=99, k=1000)')
    plt.plot(r, C, '*', label='Test set (n=149, k=1000)')

    plt.ylabel('Performance')
    plt.xlabel('Candidate')

    plt.title('Performance by candidates at training and test sets')
    plt.legend(loc='lower right', frameon=True)

    plt.show(block=True)
