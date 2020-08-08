# code notes

data: time $t$, timeseries $y$

# model components

## noise $\sigma$

`PMProphet.finalize_model`:

```python
pm.Normal(     "y_%s" % self.name,
                mu=(self.y - self.data["y"].mean()) / self.data["y"].std(),
                sd=self.priors["sigma"],
                observed=(self.data["y"] - self.data["y"].mean())
                         / self.data["y"].std(),
            )
pm.Deterministic("y_hat_%s" % self.name, self.y)
```

`self.y` is created during fit; 

`sigma` plays role of additive, Gaussian noise and has Half Cauchy prior.

$$\hat y = N(\mu_{pm}(t), \sigma^2),\\
\sigma \sim \text{HalfCauchy}$$

where $\mu_{pm}$ is inferred by the model (see below).

## intercept $d$

`PMProphet._prepare_fit:`

```python
y += self.priors["intercept"]
```

created at fit time;

$$\mu_{pm} \stackrel{+}{=} d,\\
d \sim N(\bar{y}, \sigma^2_y)$$

## growth g

# prediction

`PMProphet.predict:`

```python
y_hat_noised = np.random.normal(
            y_hat[:, self.skip_first:],
            self.data['y'].std() * self.trace[self.priors_names["sigma"]][self.skip_first:]
        )
  ddf = pd.DataFrame(
            [
                np.percentile(y_hat_noised, 50, axis=-1),
                np.percentile(y_hat_noised, alpha / 2 * 100, axis=-1),
                np.percentile(y_hat_noised, (1 - alpha / 2) * 100, axis=-1),
            ]
        ).T

ddf["ds"] = m.data["ds"]
ddf.columns = ["y_hat", "y_low", "y_high", "ds"]
```

**predict/inference: median and percentile as y_hat, y_lower/upper/**