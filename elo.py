
############################################################
# Example ELO helper. You could also do a Bradley-Terry fit.
############################################################
def update_elo(rating_a, rating_b, result_a, k=32):
    """
    rating_a, rating_b: current Elo of model A and model B
    result_a: 1 if A wins, 0 if B wins, 0.5 if tie
    k: step size
    Returns updated rating_a, rating_b
    """
    expected_a = 1.0 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1.0 - expected_a
    new_rating_a = rating_a + k * (result_a - expected_a)
    new_rating_b = rating_b + k * ((1 - result_a) - expected_b)
    return new_rating_a, new_rating_b


