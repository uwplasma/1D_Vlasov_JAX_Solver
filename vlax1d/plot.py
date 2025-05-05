import matplotlib.pyplot as plt


def plot_results(k_values, slopes, mathematica_data, m_max):
    """
    Plots the results of the slope versus k value calculation alongside its
    comparison with Mathematica data. The generated plot visualizes the
    relationship between the k values and the corresponding slopes, and it
    includes a display of Mathematica's output for validation purposes.

    :param k_values: A list or array of values for the variable k.
    :param slopes: A list or array of slope values corresponding to the k_values.
    :param mathematica_data: A DataFrame containing Mathematica's results for
        slope vs. k, where the first column represents k values and the second
        column represents the associated slopes.
    :param m_max: A numeric value representing the maximum m value used in the
        calculation, which will be displayed in the plot title.
    :return: None
    """
    plt.figure()
    plt.plot(k_values, slopes, '-o', label='Slope vs. k')
    plt.plot(mathematica_data.iloc[:, 0], mathematica_data.iloc[:, 1], '-', label='Mathematica Slope vs. k')
    plt.xlim(0.0, 0.5)
    plt.ylim(-0.05, 0.05)
    plt.xlabel('k')
    plt.ylabel('Slope')
    plt.title(f'Slope vs. k Value (m_max: {m_max})')
    plt.legend()
    plt.show()
