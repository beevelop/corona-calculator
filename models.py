import itertools

import pandas as pd

import data.constants as constants

_STATUSES_TO_SHOW = [
    "Infected",
    "Dead",
    "Need Hospitalization",
    "Susceptible",
    "Recovered",
]


def get_predictions(
    cases_estimator, sir_model, num_diagnosed, area_population, max_days
):

    true_cases = cases_estimator.predict(num_diagnosed)

    # For now assume removed starts at 0. Doesn't have a huge effect on the model
    predictions = sir_model.predict(
        susceptible=area_population - true_cases,
        infected=true_cases,
        removed=0,
        time_steps=max_days,
    )

    num_entries = max_days + 1

    # Have to use the long format to make plotly express happy
    df = pd.DataFrame(
        {
            "Days": list(range(num_entries)) * len(_STATUSES_TO_SHOW),
            "Forecast": list(
                itertools.chain.from_iterable(
                    predictions[status] for status in _STATUSES_TO_SHOW
                )
            ),
            "Status": list(
                itertools.chain.from_iterable(
                    [status] * num_entries for status in _STATUSES_TO_SHOW
                )
            ),
        }
    )
    return df


def get_deaths_by_age_group(death_prediction: int, infections_predicted: int):
    """
    Get outcomes segmented by age. The important assumption here is that age groups get infected at the same rate, that
    is every group is as likely to contract the infection.
    :param death_prediction: Number of deaths predicted.
    :param infections_predicted: Number of infections predicted.
    :return: outcome by age in a DataFrame.
    """
    age_data = constants.AgeData.data
    age_data["Deaths"] = 0

    # Effective mortality rate may be different than the one defined in data/constants.py because once we reach
    # hospital capacity, we increase the death rate.
    effective_death_rate = death_prediction / infections_predicted
    death_increase_ratio = effective_death_rate / constants.MortalityRate.default

    for ag in age_data.index:
        # Filter out current age group.
        other_age_data = age_data[age_data.index != ag]
        # Weighted mortality from other age groups.
        other_age_weighted_death_rate = (
            other_age_data.Proportion * other_age_data.Dead * death_increase_ratio
        )
        # Compute deaths from other age groups and sum.
        deaths_from_other_groups = (
            other_age_weighted_death_rate * infections_predicted
        ).sum()
        age_data.loc[ag, "Deaths"] = round(death_prediction - deaths_from_other_groups)
    return age_data.iloc[:, -1:]


class TrueInfectedCasesModel:
    """
    Used to estimate total number of true infected persons based on either number of diagnosed cases or number of deaths.
    """

    def __init__(self, ascertainment_rate):
        """
        :param ascertainment_rate: Ratio of diagnosed to true number of infected persons.
        """
        self._ascertainment_rate = ascertainment_rate

    def predict(self, diagnosed_cases):
        return diagnosed_cases / self._ascertainment_rate


class SIRModel:
    def __init__(
        self,
        transmission_rate_per_contact,
        contact_rate,
        recovery_rate,
        normal_death_rate,
        critical_death_rate,
        hospitalization_rate,
        hospital_capacity,
    ):
        """
        :param transmission_rate_per_contact: Prob of contact between infected and susceptible leading to infection.
        :param contact_rate: Mean number of daily contacts between an infected individual and susceptible people.
        :param recovery_rate: Rate of recovery of infected individuals.
        :param normal_death_rate: Average death rate in normal conditions.
        :param critical_death_rate: Rate of mortality among severe or critical cases that can't get access
            to necessary medical facilities.
        :param hospitalization_rate: Proportion of illnesses who need are severely ill and need acute medical care.
        :param hospital_capacity: Max capacity of medical system in area.
        """
        self._infection_rate = transmission_rate_per_contact * contact_rate
        self._recovery_rate = recovery_rate
        # Death rate is amortized over the recovery period
        # since the chances of dying per day are mortality rate / number of days with infection
        self._normal_death_rate = normal_death_rate * recovery_rate
        # Death rate of sever cases with no access to medical care.
        self._critical_death_rate = critical_death_rate * recovery_rate
        self._hospitalization_rate = hospitalization_rate
        self._hospital_capacity = hospital_capacity

    def predict(self, susceptible, infected, removed, time_steps=100):
        """
        Run simulation.
        :param susceptible: Number of susceptible people in population.
        :param infected: Number of infected people in population.
        :param removed: Number of recovered people in population.
        :param time_steps: Time steps to run simulation for
        :return: List of values for S, I, R over time steps
        """
        population = susceptible + infected + removed

        S = [int(susceptible)]
        I = [int(infected)]
        R = [int(removed)]
        D = [0]
        H = [round(self._hospitalization_rate * infected)]

        for t in range(time_steps):

            # There is an additional chance of dying if people are critically ill
            # and have no access to the medical system.
            if I[-1] > 0:
                underserved_critically_ill_proportion = (
                    max(0, H[-1] - self._hospital_capacity) / I[-1]
                )
            else:
                underserved_critically_ill_proportion = 0
            weighted_death_rate = (
                self._normal_death_rate * (1 - underserved_critically_ill_proportion)
                + self._critical_death_rate * underserved_critically_ill_proportion
            )

            # Forecast

            s_t = S[-1] - self._infection_rate * I[-1] * S[-1] / population
            i_t = (
                I[-1]
                + self._infection_rate * I[-1] * S[-1] / population
                - (weighted_death_rate + self._recovery_rate) * I[-1]
            )
            r_t = R[-1] + self._recovery_rate * I[-1]
            d_t = D[-1] + weighted_death_rate * I[-1]

            h_t = self._hospitalization_rate * i_t

            S.append(round(s_t))
            I.append(round(i_t))
            R.append(round(r_t))
            D.append(round(d_t))
            H.append(round(h_t))

        return {
            "Susceptible": S,
            "Infected": I,
            "Recovered": R,
            "Dead": D,
            "Need Hospitalization": H,
        }
