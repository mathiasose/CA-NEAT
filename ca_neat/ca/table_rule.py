from itertools import product


def table_rule_from_string(string, alphabet,
                           neighborhood_size):
    N = neighborhood_size

    inputs = list(product(alphabet, repeat=N))

    return {
        k: v for k, v in zip(inputs, string)
    }


if __name__ == '__main__':
    print(*table_rule_from_string(
        string='11111111111111111111111111111111111111111111111111111111111111111010101010101010111111101111101010101010101010101111111011111010',
        alphabet='01',
        neighborhood_size=7,
    ).items(), sep='\n')
