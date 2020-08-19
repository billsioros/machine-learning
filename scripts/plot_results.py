
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def iterations_with_respect_to(df, field):
    verbose = {
        'e': 'acceptable error margin',
        'k': 'number of components'
    }

    for value in df[field].unique():
        subset = df[df[field] == value]

        other_field = 'k' if field == 'e' else 'e'

        if field == 'e':
            label = f'{field} = {value:e}'
        else:
            label = f'{field} = {value:d}'

        plt.plot(subset[other_field], subset['Iterations'], label=label)

    if field == 'e':
        plt.xlim(df[other_field].min(), df[other_field].max())
    else:
        plt.xlim(df[other_field].max(), df[other_field].min())

    plt.ylim(df['Iterations'].min() - 1, df['Iterations'].max() + 1)

    plt.xscale('log')
    plt.yscale('linear')

    plt.xticks(df[other_field])

    plt.xlabel(field.title())
    plt.ylabel('Iterations')

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.title(
        f'The number of iterations '
        f'as a function of'
        f'the {verbose[other_field]}'
    )
    plt.show()


if __name__ == "__main__":
    path = Path.cwd() / 'nmf_convergence.csv'

    df = pd.read_csv(path, sep=r'\s*,\s*', engine='python')

    iterations_with_respect_to(df, 'e')
    iterations_with_respect_to(df, 'k')
