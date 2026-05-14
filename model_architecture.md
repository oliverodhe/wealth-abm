# Model Architecture: Implemented Non-OLG Swedish-Inspired Tax-Transfer ABM

This document describes the current implemented model architecture. It differs from the original intended architecture in `model_architecture.md` by focusing on a clean, fixed-population, non-OLG baseline with Swedish-inspired labour taxation, transfer-policy decompositions, return-heterogeneity robustness checks, and mobility diagnostics.

The current model is designed to answer:

> How much does revenue-neutral labour-income tax progressivity affect wealth inequality and mobility when wealth accumulation is also driven by heterogeneous capital returns and redistribution design?

---

## 1. Current Model Scope

The implemented model is a discrete-time agent-based model with:

- fixed population
- annual time steps
- heterogeneous wealth, skill, saving rates, and return classes
- persistent labour income
- bounded heterogeneous capital returns
- Swedish-inspired labour-income tax scenarios
- Swedish-inspired transfer-policy variants
- revenue-neutral first-year labour-tax calibration
- wealth inequality, income inequality, return, transfer, and mobility diagnostics

The model deliberately does **not** yet include:

- age
- OLG structure
- pensions
- inheritance
- parent-child links
- historical Swedish calibration

---

## 2. Agent State

Each agent currently has:

| State | Description |
|---|---|
| `wealth` | Net wealth, updated annually. |
| `skill` | Permanent labour-market productivity component. |
| `income_shock` | Persistent labour-income shock. |
| `saving_rate` | Agent-specific saving propensity. |
| `return_class` | Persistent capital-return class: low, medium, or high. |

The model also stores:

| State | Description |
|---|---|
| `initial_wealth` | Wealth at simulation start, used for mobility metrics. |
| `labour_income` | Current pre-tax labour income generated from skill and shock. |

Not yet implemented:

- `age`
- `cohort`
- `parent_id`
- retirement state
- unemployment history
- pension income

---

## 3. Initial Conditions

Initial wealth is skewed:

```text
90% lognormal wealth
10% Pareto top-tail wealth
rescaled to target mean wealth
```

Agents also receive:

- normally distributed permanent skill
- persistent initial income shock
- heterogeneous saving rates
- persistent return class

The current initial distribution is synthetic. It is not calibrated to Swedish data.

---

## 4. Labour-Income Process

Labour income is persistent:

```text
income_i,t = base_income * exp(skill_i + shock_i,t)
shock_i,t = rho * shock_i,t-1 + epsilon_i,t
```

where `epsilon` is normally distributed.

Unemployment is implemented as a simple annual shock:

```text
unemployed_i,t ~ Bernoulli(unemployment_probability)
if unemployed:
    taxable labour income = 0
```

Unemployed agents may receive replacement income depending on the transfer policy.

Not yet implemented:

- age-income profile
- sectoral labour markets
- endogenous unemployment
- persistent unemployment spells
- wage growth by cohort

---

## 5. Capital-Return Process

Each agent belongs to a persistent return class:

| Class | Baseline mean |
|---|---:|
| Low | 0.02 |
| Medium | 0.04 |
| High | 0.06 |

Annual return:

```text
r_i,t = class_mean_i + normal_shock
```

Returns are bounded:

```text
min_capital_return <= r_i,t <= max_capital_return
```

Capital returns do **not** depend directly on wealth.

Robustness presets vary return-class means and return shock standard deviation:

| Preset | Purpose |
|---|---|
| `low_return_heterogeneity` | Capital returns are relatively similar across agents. |
| `baseline_return_heterogeneity` | Baseline class spread. |
| `high_return_heterogeneity` | Wealth accumulation is more strongly driven by capital-return differences. |

---

## 6. Wealth Law Of Motion

Each year:

```text
capital_income_i,t = wealth_i,t * r_i,t
capital_tax_i,t = max(capital_income_i,t, 0) * capital_tax_rate
after_tax_capital_income_i,t = capital_income_i,t - capital_tax_i,t

disposable_income_i,t = labour_income_i,t - labour_tax_i,t + transfers_i,t
saving_i,t = max(0, saving_rate_i * (disposable_income_i,t - min_consumption))

wealth_i,t+1 = max(0, wealth_i,t + after_tax_capital_income_i,t + saving_i,t)
```

This keeps the main mechanism explicit:

- labour taxation affects new saving flows
- capital returns affect existing wealth
- transfer design affects disposable income and saving capacity

---

## 7. Labour-Tax Systems

The model supports:

| Tax system | Description |
|---|---|
| `flat` | Flat labour-income tax. |
| `simple_progressive` | Earlier simple threshold tax, retained for compatibility. |
| `swedish` | Municipal + state tax + earned-income credit approximation. |

The Swedish-style formula is:

```text
T(y) = municipal_rate * y
     + state_rate * max(0, y - state_threshold)
     - earned_income_credit(y)
```

The earned-income credit:

- increases disposable income for low/middle earners
- phases out for high earners
- never exceeds pre-credit tax liability

---

## 8. Tax Scenarios

Current tax scenarios:

| Scenario | Description |
|---|---|
| `flat` | Flat labour-income tax. |
| `swedish_low_progressivity` | Higher threshold, lower state rate, smaller credit phaseout. |
| `swedish_baseline_progressivity` | Baseline Swedish-style approximation. |
| `swedish_high_progressivity` | Lower threshold, higher state rate, stronger credit phaseout. |

For Swedish scenarios, the municipal rate is calibrated so first-year labour-tax revenue matches the flat-tax target.

---

## 9. Revenue Neutrality

The model uses first-year labour-income revenue neutrality:

```text
target_revenue = flat_tax(first_year_labour_income).sum()
```

For each Swedish-style tax scenario:

```text
calibrate municipal_rate
such that Swedish tax revenue ~= target_revenue
```

Capital-tax revenue is recorded separately and is not used for revenue-neutral labour-tax calibration.

---

## 10. Transfer System

Transfers are funded from **labour-tax revenue only**.

Capital-tax revenue is recorded separately and is not redistributed.

Implemented transfer components:

### 10.1 Universal Transfer

Equal payment to all agents.

### 10.2 Means-Tested Safety Floor

If disposable income after labour tax falls below a threshold:

```text
means_tested_transfer = replacement_rate * max(0, safety_floor - disposable_income_after_tax)
```

### 10.3 Unemployment Support

If an agent receives an unemployment shock:

```text
unemployment_support = min(replacement_rate * previous_income, benefit_cap)
```

### 10.4 Budget Handling

The transfer system is budget-balanced against labour-tax revenue.

If targeted transfers exceed the targeted budget, they are scaled down. Remaining revenue is paid through the universal component.

---

## 11. Transfer Policies

Current transfer-policy decompositions:

| Policy | Description |
|---|---|
| `no_transfers` | No transfer spending. |
| `lump_sum_only` | All labour-tax revenue redistributed equally. |
| `universal_plus_safety_floor` | Universal transfer plus means-tested safety floor. |

The default model also supports unemployment support through the broader Swedish transfer logic, but the transfer-policy decomposition isolates the three policies above.

Policy invariants:

- `no_transfers`: transfer spending share = 0, means-tested recipient share = 0
- `lump_sum_only`: means-tested recipient share = 0
- `universal_plus_safety_floor`: means-tested recipient share may be positive

---

## 12. Decomposition Toggles

The model includes independent mechanism toggles:

| Toggle | Effect |
|---|---|
| `no_transfers` | Removes all transfer spending. |
| `homogeneous_returns` | Removes return-class differences. |
| `homogeneous_saving_rates` | Gives all agents the same saving rate. |
| `no_capital_tax` | Sets capital tax to zero. |

These are used in decomposition experiments to identify mechanisms driving inequality and mobility.

---

## 13. Experiment Runners

### 13.1 `scripts/run_scenarios.py`

Runs:

```text
3 return presets x 4 tax scenarios = 12 runs
```

Outputs:

- yearly CSVs
- combined scenario comparison CSV
- final summary CSV
- mobility summary CSV
- return heterogeneity summary CSV
- inequality and Lorenz plots

### 13.2 `scripts/run_decompositions.py`

Runs mechanism decomposition toggles for:

```text
flat
swedish_baseline_progressivity
swedish_high_progressivity
```

Outputs:

- `decomposition_comparison.csv`
- `decomposition_final_summary.csv`
- `decomposition_mobility_summary.csv`
- `decomposition_summary.csv`
- decomposition plots

### 13.3 `scripts/run_transfer_policy_decompositions.py`

Runs the full transfer-policy grid:

```text
3 return presets x 3 transfer policies x 4 tax scenarios = 36 runs
```

Each printed block has the format:

```text
{return_preset} / {transfer_policy} / {tax_scenario}
```

Outputs:

- `transfer_policy_yearly_results.csv`
- `transfer_policy_final_summary.csv`
- `transfer_policy_mobility_summary.csv`
- `decomposition_summary.csv`
- `transfer_policy_decomposition_summary.csv`
- transfer-policy comparison plots

---

## 14. Recorded Yearly Metrics

Each yearly result includes:

### Wealth Metrics

- wealth Gini
- top 10% wealth share
- top 1% wealth share

### Tax And Transfer Metrics

- labour-tax revenue
- capital-tax revenue
- total tax revenue
- total transfers
- universal transfer spending
- means-tested transfer spending
- unemployment transfer spending
- transfer spending share
- means-tested recipient count
- means-tested recipient share
- unemployment rate

### Income Metrics

- pre-tax labour-income Gini
- disposable-income Gini
- capital-income Gini
- capital income share of total income
- average tax rate by income decile

### Return Diagnostics

- average realised returns by wealth decile
- average realised returns by return class

---

## 15. Mobility Metrics

The model computes:

- wealth-quintile transition matrix
- Shorrocks mobility index
- initial-final wealth rank correlation
- probability of remaining in top 20%
- probability of moving from bottom 40% to top 40%

Mobility is measured from initial wealth to final wealth within the fixed population.

---

## 16. Current Output Tables

Important CSV outputs include:

| File | Description |
|---|---|
| `scenario_comparison.csv` | Main scenario panel. |
| `final_summary.csv` | Final-year scenario outcomes. |
| `mobility_summary.csv` | Mobility outcomes. |
| `return_heterogeneity_summary.csv` | Differences between flat and high progressivity by return preset. |
| `decomposition_summary.csv` | Decomposition effects. |
| `transfer_policy_final_summary.csv` | Full 36-run transfer-policy grid final outcomes. |
| `transfer_policy_mobility_summary.csv` | Mobility outcomes for transfer-policy grid. |
| `transfer_policy_decomposition_summary.csv` | Differences relative to flat by transfer policy and return preset. |

---

## 17. Current Plots

The model saves plots for:

- wealth Gini over time
- pre-tax labour-income Gini over time
- disposable-income Gini over time
- capital income share over time
- top 10% wealth share over time
- top 1% wealth share over time
- final Lorenz curves
- decomposition comparisons
- transfer-policy comparisons:
  - wealth Gini by transfer policy
  - disposable-income Gini by transfer policy
  - top 1% share by transfer policy

---

## 18. What This Version Can Answer

The implemented architecture can study:

1. Whether Swedish-style labour-tax progressivity compresses disposable income.
2. Whether that disposable-income compression translates into lower wealth inequality.
3. Whether capital-return heterogeneity weakens the wealth effect of labour-tax progressivity.
4. Whether redistribution design matters more than labour-tax progressivity.
5. Whether top wealth persistence changes under different tax-transfer designs.
6. Whether mobility responds differently from inequality.

---

## 19. Differences From The Intended Final Architecture

The current model is richer than the original Stage 1/2 prototype, but it is still not the full intended model.

### 19.1 Missing OLG And Demographics

Not implemented:

- age structure
- retirement
- pensions
- death
- offspring replacement
- cohort identifiers

### 19.2 Missing Inheritance

Not implemented:

- parent-child links
- bequests
- inheritance taxation
- intergenerational wealth persistence
- parent-child rank correlations

### 19.3 No Historical Calibration

The model is not calibrated to Swedish historical data.

Missing calibration targets include:

- Swedish wealth Gini
- top 10% and top 1% wealth shares
- income distribution
- tax revenue as share of income
- average tax burden
- capital-income share
- Pareto tail behaviour

### 19.4 Labour Market Still Simple

Implemented unemployment is a simple iid annual shock.

Not implemented:

- persistent unemployment spells
- social-insurance eligibility histories
- wage growth
- sectoral shocks
- age-income profiles

### 19.5 Transfer System Is Stylised

The transfer system is Swedish-inspired but not institutionally calibrated.

Not implemented:

- housing allowances
- child benefits
- sickness insurance
- detailed unemployment insurance rules
- municipal variation
- household composition

### 19.6 No Public Investment Channel

The original intended architecture included a possible public investment or mobility channel:

```text
public investment -> future productivity growth
```

This is not implemented.

### 19.7 No HSV Tax Robustness Function

The intended architecture suggested retaining the HSV smooth tax function as a robustness check.

Current model has:

- flat tax
- simple progressive tax
- Swedish-style tax

But no HSV tax function.

### 19.8 No Wealth-Dependent Return Premium

Capital returns are heterogeneous by persistent class, but do not directly depend on wealth.

The intended architecture suggested a possible robustness check:

```text
higher wealth -> better diversification / higher expected return
```

This is not implemented.

---

## 20. Recommended Next Steps

The next development steps should be:

1. Clean output naming so main scenario, return-robustness, mechanism decomposition, and transfer-policy decomposition outputs do not overwrite each other.
2. Add Pareto tail index and income top-share diagnostics.
3. Add a calibration module with synthetic targets first, then Swedish data targets.
4. Add richer transfer-policy parameters without adding full OLG.
5. Add HSV tax function as a robustness experiment.
6. Add optional wealth-dependent return premium as a robustness experiment.
7. Only after the non-OLG model is stable, add age, pensions, inheritance, and OLG dynamics.

