import generators
from sklearn import datasets
import seaborn as sns

for label, synthesiser in generators.all.items():
    classification = issubclass(
        generators.dist_base.ClassificationSynthesiser,
        synthesiser)
    if classification:
        for X, y in [
            datasets.make_blobs(n_features=100),
            datasets.make_classification(),
        ]:
            sampler = synthesiser()
            sampler.fit(X, y)
            sample = sampler.sample(10000)

    else:
        for X, y in [
            datasets.make_friedman1(),
            datasets.make_friedman2(),
            datasets.make_friedman3(),
        ]:
            sampler = synthesiser()
            sampler.fit(X, y)
            sample = sampler.sample(10000)

for label, synthesiser in generators.all.items():
    classification = issubclass(
        generators.dist_base.ClassificationSynthesiser,
        synthesiser)
    if not classification:
        data = sns.load_dataset('flights')
        X, y = data.drop('passengers', axis=1), data.passengers
        discrete_columns = ['month']
        sampler = synthesiser(discrete_columns=discrete_columns)
        sampler.fit(X, y)
        sample, _ = sampler.sample(10000)
        assert sample.shape[1] == X.shape[1]

    else:
        data = sns.load_dataset('exercise').drop('Unnamed: 0', axis=1)
        X, y = data.drop(['Unnamed: 0', 'id', 'kind'], axis=1), data.kind
        discrete_columns = ['diet', 'time']
        sampler = synthesiser(discrete_columns=discrete_columns)
        sampler.fit(X, y)
        sample, _ = sampler.sample(10000)
        assert sample.shape[1] == X.shape[1]
