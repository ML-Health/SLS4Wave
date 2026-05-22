import torch

samples, alarms, ys = torch.load('dataset/samples_alarms.pt')

mu = []
sigma = []

# for each signal
for i in range(11):
    vals = []
    for x in range(len(samples)):
        sample = samples[x, i]
        # if current signal is available
        if sample.sum() != 0.:
            vals.append(sample)

    vals = torch.cat(vals)

    mu.append(vals.mean())
    sigma.append(vals.std())

print(mu)
print(sigma)


for i in range(11):
    mu_i = mu[i]
    sigma_i = sigma[i]

    for x in range(len(samples)):
        
        if samples[x, i].sum() != 0.:
            # standardization using z-score
            samples[x, i] = (samples[x, i]-mu_i) / sigma_i
mu = torch.Tensor(mu)
sigma = torch.Tensor(sigma)

print(mu)
print(sigma)

# save standardized dataset
torch.save((mu, sigma, samples, alarms, ys),
           'dataset/standardized_samples_alarms.pt')
