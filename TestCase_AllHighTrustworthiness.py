import numpy as np
import matplotlib.pyplot as plt

def calculate_rins(DW, NFR, WS, A, B, C):
    """
    Calculate Reliability Metrics (Rins)
    :param DW: List of disengaged workforce values over time
    :param NFR: List of fulfilling assigned responsibilities over time
    :param WS: List of work satisfaction scores over time
    :param A: Weight for DW
    :param B: Weight for NFR
    :param C: Weight for WS
    :return: Rins value
    """
    Rins = A * np.sum(DW) + B * np.sum(NFR) + C * np.sum(WS)
    return Rins

def calculate_smtins(RH, MF, TL, X, Y, Z):
    """
    Calculate Social Media Trustworthiness Metrics (SMTins)
    :param RH: List of relative happiness values over time
    :param MF: List of motive force scores over time
    :param TL: List of trustworthiness lexicon scores over time
    :param X: Weight for RH
    :param Y: Weight for MF
    :param Z: Weight for TL
    :return: SMTins value
    """
    SMTins = X * np.sum(RH) + Y * np.sum(MF) + Z * np.sum(TL)
    return SMTins

def calculate_pins(DW, NFR, WS, RH, MF, TL, weights_org, weights_sm):
    """
    Calculate the final probability of insider threat (Pins)
    :param DW, NFR, WS: Organizational parameters lists
    :param RH, MF, TL: Social media parameters lists
    :param weights_org: Tuple (A, B, C) for organizational weights
    :param weights_sm: Tuple (X, Y, Z) for social media weights
    :return: Pins value
    """
    A, B, C = weights_org
    X, Y, Z = weights_sm
    
    Rins = calculate_rins(DW, NFR, WS, A, B, C)
    SMTins = calculate_smtins(RH, MF, TL, X, Y, Z)
    
    Pins = Rins + SMTins
    return Pins

# Example usage:
n = 10  # Time steps

# Edge Case: All High Trustworthiness Values
DW = [0] * n  # No disengagement
NFR = [9] * n  # Fully fulfilling responsibilities
WS = [10] * n  # Maximum work satisfaction
RH = [10] * n  # Maximum happiness
MF = [10] * n  # Maximum motivation
TL = [10] * n  # Maximum trustworthiness

# Define weight coefficients
weights_org = (0.5, 0.3, 0.2)  # Weights for DW, NFR, WS
weights_sm = (0.4, 0.4, 0.2)  # Weights for RH, MF, TL

# Calculate insider threat probability
Pins_value = calculate_pins(DW, NFR, WS, RH, MF, TL, weights_org, weights_sm)
print("Probability of Insider Threat (Pins) for High Trustworthiness Case:", Pins_value)

# Data Visualization
plt.figure(figsize=(10, 5))
plt.plot(range(1, n+1), DW, marker='o', linestyle='-', label='Disengaged Workforce')
plt.plot(range(1, n+1), NFR, marker='s', linestyle='-', label='Fulfilling Responsibilities')
plt.plot(range(1, n+1), WS, marker='^', linestyle='-', label='Work Satisfaction')
plt.plot(range(1, n+1), RH, marker='D', linestyle='-', label='Relative Happiness')
plt.plot(range(1, n+1), MF, marker='v', linestyle='-', label='Motive Force')
plt.plot(range(1, n+1), TL, marker='x', linestyle='-', label='Trustworthiness Lexicon')

plt.xlabel("Time Steps")
plt.ylabel("Parameter Values")
plt.title("Visualization of TMV Parameters Over Time - High Trustworthiness Case")
plt.legend()
plt.grid(True)
plt.show()
