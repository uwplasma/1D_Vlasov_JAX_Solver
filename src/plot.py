import matplotlib.pyplot as plt


def plot_results(k_values, slopes, mathematica_data, m_max):
    """
    Plot slope vs. k.

    :param k_values: Wave number k
    :param slopes: Calculated slopes
    :param mathematica_data: Reference data
    :param m_max: Maximum number of modes for all
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
