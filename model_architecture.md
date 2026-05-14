# Model Architecture: Swedish-Inspired Progressive Tax ABM

The model is designed to answer the main research question:

> **How does progressive labour-income taxation affect wealth inequality?**

A secondary objective is to study how the same tax progressivity affects social mobility. A third objective is to calibrate or validate the model against Swedish historical stylised facts before using it for counterfactual experiments.

---

## 1. Design philosophy

The model should be realistic enough to represent the main channels of wealth accumulation, but simple enough to support clear interpretation.

The core idea is:

```text
heterogeneous households
+ labour income
+ progressive labour-income taxation
+ Swedish-style transfers
+ savings and consumption
+ capital returns
+ wealth accumulation
+ inequality and mobility measurement
```

The preferred identification logic is:

```text
same agents
same income process
same capital-return process
same Swedish-style redistribution rules
same approximate total tax revenue
but different progressivity of labour-income taxation
```

This makes the interpretation cleaner. If wealth inequality changes, the change can be attributed mainly to the progressivity of labour-income taxation, not to differences in total revenue, tax base, or redistribution generosity.

---

## 2. What is kept from the included architecture

The included architecture contains several strong components that should be preserved:

1. A transparent wealth law of motion.
2. Heterogeneous agents.
3. Persistent labour-income differences.
4. Heterogeneous capital returns.
5. A consumption and saving rule.
6. A distinction between labour income, capital income, wealth, and transfers.
7. Inequality metrics such as Gini, top wealth shares, and Lorenz curves.
8. Mobility metrics such as wealth-rank transition matrices.
9. Revenue-neutral comparisons between flat and progressive labour taxation.
10. Optional intergenerational mechanisms for later robustness checks.

---

## 3. Model overview

The model is a discrete-time agent-based model with a fixed population of heterogeneous households.

Each period represents one year. Agents earn labour income, pay taxes, receive transfers, consume, save, earn capital returns, and update wealth. The government collects labour-income taxes and redistributes part of the revenue through a simplified Swedish-style transfer system.

The model has no firms, banks, housing market, or general-equilibrium price clearing. Wages, capital returns, and demographic processes are exogenous. This keeps the model focused on the link between taxation and wealth accumulation.

---

## 4. Agent state

Each agent represents a household or individual. The baseline state variables are:

| Variable | Description |
|---|---|
| `age_i` | Age of agent. Used for life-cycle income, retirement, and optional inheritance. |
| `wealth_i` | Net wealth. Main state variable and basis for inequality measurement. |
| `skill_i` | Persistent labour-market ability or productivity. |
| `income_i` | Annual pre-tax labour income. |
| `saving_rate_i` | Agent-specific tendency to save out of disposable income. |
| `return_type_i` | Persistent investment-return type. Used to generate heterogeneous capital returns. |
| `capital_income_i` | Annual return on existing wealth. |
| `tax_paid_i` | Labour-income tax paid in the current period. |
| `transfer_i` | Transfers received in the current period. |
| `cohort_i` | Optional generation/cohort identifier. Needed only if intergenerational mobility is included. |
| `parent_id_i` | Optional link to parent. Needed only if inheritance or parent-child mobility is included. |

A minimal first implementation can omit `cohort_i` and `parent_id_i`. They should be added once the baseline model works.

---

## 5. Annual cycle

Each simulated year follows this schedule:

```text
1. Age agents and handle retirement/death if demographic module is active.
2. Generate labour income.
3. Generate capital income from existing wealth.
4. Apply labour-income tax.
5. Apply capital-income tax if included.
6. Redistribute revenue through Swedish-style transfer rules.
7. Agents consume and save.
8. Update wealth.
9. Update productivity if public investment / mobility channel is active.
10. Record inequality and mobility outcomes.
```

The first version can use a simpler schedule without deaths and births:

```text
1. Generate labour income.
2. Generate capital income.
3. Tax labour income.
4. Redistribute transfers.
5. Consume and save.
6. Update wealth.
7. Record outcomes.
```

The OLG version is more realistic, but the non-OLG version is easier to debug and should be implemented first.

---

## 6. Wealth law of motion

The core wealth update is:

$$
w_{i,t+1} = w_{i,t} + s_{i,t} + (1 - \tau_k) r_{i,t} w_{i,t}
$$

where:

- $w_{i,t}$ is wealth,
- $s_{i,t}$ is saving from disposable income,
- $r_{i,t}$ is the capital return,
- $\tau_k$ is the capital-income tax rate.

Saving is determined by disposable income after taxes and transfers:

$$
s_{i,t} = \max\left(0, \sigma_i \cdot (y^{disp}_{i,t} - c^{min}_{i,t})\right)
$$

where:

$$
y^{disp}_{i,t} = y_{i,t} - T(y_{i,t}) + Tr_{i,t}
$$

and:

- $y_{i,t}$ is pre-tax labour income,
- $T(y_{i,t})$ is labour-income tax,
- $Tr_{i,t}$ is transfers,
- $c^{min}_{i,t}$ is minimum consumption,
- $\sigma_i$ is the saving propensity.

This equation makes the main mechanism clear:

```text
Progressive labour-income taxation can reduce new saving flows,
especially among high-income agents.

But existing wealth can still grow through capital returns,
which may limit the effect of labour-income taxation on wealth inequality.
```

---

## 7. Initial wealth distribution

The model should not start from equal wealth. Initial wealth should be drawn from a skewed distribution.

Recommended baseline:

```text
bottom / middle of distribution: lognormal wealth
top tail: Pareto wealth
```

Implementation idea:

```python
n_bottom = int(0.90 * N)
n_top = N - n_bottom

bottom_wealth = lognormal_distribution(n_bottom)
top_wealth = pareto_distribution(n_top)

wealth = concatenate(bottom_wealth, top_wealth)
wealth = rescale_to_target_mean(wealth)
```

The distribution should be calibrated so that the model starts from a plausible Swedish or Sweden-like wealth distribution. Calibration targets can include:

```text
wealth Gini
top 10% wealth share
top 1% wealth share
Lorenz curve shape
Pareto tail behaviour
```

---

## 8. Labour-income process

Labour income should be persistent, not independently redrawn each year.

Recommended specification:

$$
\log y_{i,t} = \phi(age_{i,t}) + skill_i + u_{i,t}
$$

with persistent shocks:

$$
u_{i,t} = \rho u_{i,t-1} + \epsilon_{i,t}, \qquad \epsilon_{i,t} \sim N(0, \sigma_\epsilon^2)
$$

where:

- $skill_i$ is permanent earning ability,
- $u_{i,t}$ is persistent income variation,
- $\phi(age)$ is a hump-shaped age-income profile,
- $\rho$ controls income persistence.

A simple first version can use:

```text
income_i,t = base_income × exp(skill_i + shock_i,t)
```

A better version adds:

```text
age-income profile
unemployment shocks
retirement and pensions
```

---

## 9. Capital-return process

Capital returns are essential because the thesis studies wealth inequality, not only income inequality.

Recommended specification:

$$
r_{i,t} = \bar r + \eta_{type(i),t} + \epsilon^r_{i,t}
$$

where:

- $\bar r$ is the average return,
- $\eta_{type(i),t}$ depends on the agent's persistent return type,
- $\epsilon^r_{i,t}$ is an idiosyncratic return shock.

A realistic addition is to allow wealthier agents to have access to slightly higher or less volatile returns:

```text
higher wealth → better diversification / higher expected return
```

However, this should be added carefully. If the return advantage is too strong, the model may mechanically generate extreme inequality regardless of tax policy.

Recommended first implementation:

```text
heterogeneous but bounded return types
no direct wealth-dependent return premium at first
```

Recommended robustness check:

```text
activate wealth-dependent return premium
compare whether progressive labour tax becomes less effective
```

---

## 10. Consumption and saving

The model should include a simple behavioural saving rule rather than solving a full utility-maximisation problem.

Recommended rule:

```text
minimum consumption is paid first
remaining disposable income is partly saved
saving rate increases weakly with income or wealth
```

Example:

$$
c_{i,t} = c^{min}_{i,t} + \alpha_i y^{disp}_{i,t}
$$

or equivalently:

$$
s_{i,t} = \sigma_i \max(y^{disp}_{i,t} - c^{min}_{i,t}, 0)
$$

where $\sigma_i$ may depend on the agent's income or wealth rank.

This is important because progressive labour-income taxation affects wealth mainly through saving flows. If high-income agents save a larger share of income, then taxing high labour incomes can reduce wealth accumulation more than a flat tax with the same revenue.

---

## 11. Swedish-inspired labour-income tax module

The central policy lever is the progressivity of labour-income taxation.

The model should support two tax specifications:

## 11.1 Realistic Swedish-style bracket approximation

This is preferred for the main Swedish model.

```text
municipal tax component:
    approximately flat tax on labour income

state tax component:
    additional tax above a high-income threshold

earned-income tax credit:
    simplified reduction for labour income
```

A simplified formula:

$$
T(y_i) = \tau_m y_i + \tau_s \max(0, y_i - \bar y_s) - J(y_i)
$$

where:

- $\tau_m$ is the municipal tax rate,
- $\tau_s$ is the state tax rate above threshold,
- $\bar y_s$ is the state-tax threshold,
- $J(y_i)$ is a simplified earned-income tax credit.

Progressivity can be varied by changing:

```text
state tax threshold
state tax rate
earned-income tax credit shape
```

## 11.2 Smooth HSV tax function

The HSV function should be retained as an alternative because it makes progressivity easy to vary smoothly:

$$
T(y) = y - \lambda y^{1 - \tau_p}
$$

where:

- $\tau_p$ controls progressivity,
- $\lambda$ controls the average tax level.

This is useful for clean comparative statics and robustness checks.

## Recommendation

Use the Swedish bracket approximation in the main results, and use the HSV function as a robustness check.

---

## 12. Revenue neutrality

The main experiments should be revenue-neutral.

For each tax-progressivity scenario, choose the tax-level parameter so that total labour-income tax revenue is approximately equal across scenarios.

```text
flat tax revenue ≈ moderate progressivity revenue ≈ high progressivity revenue
```

This is crucial because otherwise the experiment confounds:

```text
progressivity effect
with
total revenue effect
```

Implementation:

```python
for each progressivity_setting:
    find tax_level_parameter such that
        total_tax_revenue ≈ target_revenue
```

For a bracket schedule, this can mean adjusting the municipal component or a scaling factor. For the HSV function, this means solving for `lambda` given the chosen `tau_p`.

---

## 13. Swedish-style redistribution module

The redistribution system should remain mostly fixed across progressivity scenarios.

A simplified Swedish-style redistribution block can include:

```text
universal component
social insurance component
means-tested safety floor
public investment / mobility component
```

## 13.1 Universal component

A fixed lump-sum transfer or household allowance.

Simplified version:

```text
universal_transfer_i = fixed amount
```

## 13.2 Social insurance component

Agents hit by unemployment or income shocks receive partial replacement income.

```text
if unemployed:
    benefit = min(replacement_rate × previous_income, benefit_cap)
```

## 13.3 Means-tested safety floor

Agents whose disposable resources fall below a minimum standard receive a top-up.

```text
social_assistance_i = max(0, minimum_standard_i - disposable_income_i)
```

## 13.4 Public investment / mobility channel

A share of revenue can be used to increase future productivity, especially for low-income or low-wealth agents.

```text
public_investment_i,t → higher expected productivity_i,t+1
```

This channel should be optional in the first version. It is more relevant for social mobility than for the headline wealth-inequality result.

---

## 14. Experimental design

Recommended main scenarios:

| Scenario | Labour-income tax | Redistribution | Revenue |
|---|---|---|---|
| A — Flat tax | Flat labour-income tax | Swedish-style transfers | Target revenue |
| B — Swedish-style progressivity | Municipal + state tax approximation | Same transfers | Same target revenue |
| C — High progressivity | Stronger state tax / lower threshold | Same transfers | Same target revenue |
| D — Low progressivity | Weaker state tax / higher threshold | Same transfers | Same target revenue |

The primary comparison is:

```text
B - A
```

That is, Swedish-style progressive labour-income taxation versus a flat labour-income tax that raises the same revenue.

The other comparisons are sensitivity checks:

```text
C - B: effect of stronger progressivity
D - B: effect of weaker progressivity
```

---

## 15. Optional decomposition experiments

After the main results, the model can run decomposition experiments. These should not replace the main experiment.

Possible decompositions:

| Decomposition | Purpose |
|---|---|
| Remove transfers | Separates taxation from redistribution. |
| Remove capital-income tax | Tests whether labour-income progressivity alone matters. |
| Remove public investment channel | Tests whether mobility effects come from productivity changes. |
| Remove heterogeneous capital returns | Tests whether capital returns weaken labour-tax effects. |
| Remove inheritance | Tests whether intergenerational persistence matters. |

These are useful because they explain why progressive labour-income taxation does or does not affect wealth inequality.

---

## 16. Historical calibration to Sweden

The model should be calibrated or validated against Swedish historical stylised facts before counterfactual experiments are interpreted.

The goal is not to reproduce Sweden year by year. The goal is to match broad patterns such as:

```text
wealth is more unequal than income
wealth has a heavy upper tail
top wealth shares are persistent
income taxes and transfers reduce disposable-income inequality
wealth inequality responds slowly to income-tax changes
capital returns matter strongly for top wealth accumulation
```

Possible calibration targets:

```text
wealth Gini
top 10% wealth share
top 1% wealth share
income Gini before and after taxes/transfers
total tax revenue as share of income
average labour-income tax burden
capital-income share of total income
```

Recommended calibration strategy:

```text
1. Choose a historical Swedish calibration window.
2. Initialise wealth and income distributions to match the start of the window.
3. Tune income persistence, saving rates, and return heterogeneity.
4. Check whether simulated inequality trends resemble Swedish stylised facts.
5. Freeze calibrated parameters.
6. Run counterfactual progressivity experiments.
```

This makes the model more credible without claiming full historical reconstruction.

---

## 17. Outcome measurement

The primary outcomes are wealth-inequality measures:

```text
wealth Gini
top 10% wealth share
top 1% wealth share
Lorenz curve
Pareto tail index
```

Secondary outcomes are social-mobility measures:

```text
wealth-quintile transition matrix
Shorrocks mobility index
rank-rank correlation over time
probability of moving from bottom 40% to middle/top groups
probability of remaining in top 20%
```

If the OLG module is active, add:

```text
parent-child wealth-rank correlation
intergenerational elasticity of wealth
inheritance share of wealth
```

The main thesis figures should show:

```text
Gini over time by tax scenario
top 10% wealth share over time by tax scenario
Lorenz curves at selected years
mobility transition matrices
sensitivity to capital-return heterogeneity
```

---

## 18. Implementation order

The implementation should be staged.

## Stage 1 — Minimal working model

```text
N heterogeneous agents
initial skewed wealth distribution
labour income process
capital returns
saving rule
flat vs progressive labour tax
fixed lump-sum redistribution
Gini and top-share outputs
```

Goal: test whether progressivity affects wealth inequality in the simplest Swedish-inspired setting.

## Stage 2 — Swedish tax-transfer approximation

```text
municipal tax component
state tax threshold
simplified earned-income tax credit
social insurance shock
means-tested safety floor
revenue-neutral calibration
```

Goal: make the model institutionally Swedish.

## Stage 3 — Historical calibration

```text
calibrate initial wealth distribution
calibrate income distribution
calibrate return heterogeneity
calibrate tax revenue / average tax burden
validate against Swedish stylised facts
```

Goal: strengthen credibility of counterfactual results.

## Stage 4 — Mobility and OLG extension

```text
age structure
retirement and pensions
death and inheritance
offspring replacement
parent-child rank correlations
```

Goal: add intergenerational social mobility if time allows.

## Stage 5 — Robustness checks

```text
HSV tax function
stronger/weaker capital-return heterogeneity
with/without inheritance
with/without public investment channel
with/without capital-income tax
```

Goal: explain mechanisms and limitations.

---

## 19. Recommended baseline model for the thesis

The recommended final baseline is:

```text
Agent-based model with:
    N = 5,000 households
    annual time steps
    skewed initial wealth distribution
    persistent labour income
    heterogeneous saving rates
    heterogeneous capital returns
    Swedish-style labour-income tax approximation
    Swedish-style transfers
    revenue-neutral progressivity scenarios
    wealth inequality as primary outcome
    mobility as secondary outcome
```

The main policy scenarios are:

```text
A. Flat labour-income tax
B. Swedish-style progressive labour-income tax
C. More progressive labour-income tax
D. Less progressive labour-income tax
```

All scenarios should use the same redistribution rules and approximately the same total tax revenue.

---

## 20. What is genuinely new in this combined version

The contribution is not that the model contains many different fiscal systems. The contribution is that it isolates one specific mechanism in a realistic institutional setting:

```text
How much can progressive labour-income taxation affect wealth inequality
when wealth accumulation is driven by both labour-income saving flows
and capital returns on existing wealth?
```

The model combines:

1. A Swedish-inspired tax-transfer setting.
2. A clean progressivity experiment with revenue-neutral comparison.
3. Heterogeneous capital returns so that the wealth tail can emerge realistically.
4. Historical calibration to Swedish stylised facts.
5. Wealth inequality as the primary outcome and social mobility as a secondary outcome.

This is a stronger and more focused design than the four-system architecture, while preserving the best technical parts of the literature-based model.

